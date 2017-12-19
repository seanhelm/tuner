import configparser
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA


def pca_groups(df, name, ncomp):
    """
    Use PCA to combine time series features into ncomp features

    :param df: DataFrame of time series features (should be one type (e.g. MFCC))
    :param name: Name to use for the feature prefix
    :param ncomp: Number of features to return per feature type
    :returns: New DataFrame with PCA-transformed features
    """
    cols = []
    arr = []

    # Iterate through each audio feature level (e.g. mean, median)
    for col in df.columns.levels[0]:
        pca_df = PCA(n_components=ncomp).fit_transform(df[col])
        
        # If split is greater than one, add a split index
        # Otherwise, just append the level name
        if ncomp > 1:
            for i in range(ncomp):
                cols.append("%s_%s_%d" % (name, col, i))
        else:
            cols.append("%s_%s" % (name, col))
            
        arr.append(pca_df)

    # Return dataframe for PCA
    return pd.DataFrame(np.hstack(arr), index=df.index, columns=cols)


def drop_level(df, name):
    """
    Drop MultiIndex levels into one level

    :param df: DataFrame to transform into one level
    :param name: Name to use for the feature prefix
    :returns: Transformed DataFrame with one level
    """
    # Drop MultiIndex level into one
    df.columns = ['_'.join(col) for col in df.columns]
    df.columns = [name + '_' + col for col in df.columns]
    return df


def filter_by_genre(genres, count):
    """
    Filter the genres by frequency count

    :param genres: Series with genre for each row
    :param count: Minimum frequency count
    :returns: New Series with genres below frequency count filtered out
    """
    # Filter out genres that appear less than count
    counts = genres.value_counts()
    counts = counts[counts >= count]
    return genres[genres.isin(counts.index)]


def select_features(features, tracks, ncomp):
    """
    Perform feature selection for the dataset

    :param features: Audio features extracted using Librosa
    :param tracks: Track metadata
    :ncomp: Ncomp taken from config, used for pca_groups method
    :returns: DataFrame with filtered dataset
    """
    # Select tracks with genre specified for training purposes
    tracks = tracks[pd.notnull(tracks['track']['genre_top'])]
    
    # Select audio features and perform PCA
    mfcc = pca_groups(features['mfcc'], "mfcc", ncomp=ncomp)
    chroma = pca_groups(features['chroma_cens'], "chroma", ncomp=ncomp)
    tonnetz = pca_groups(features['tonnetz'], "tonnetz", ncomp=ncomp)
    spectral_contrast = pca_groups(features['spectral_contrast'], "spectral_contrast", ncomp=ncomp)

    # Features to use + genres
    features = [mfcc, chroma, tonnetz, spectral_contrast]
    
    # Filter out tracks with a genre frequency of less than 1000
    genres = tracks['track']['genre_top']
    genres = filter_by_genre(genres, 1000)

    # Merge tables for audio features and genres for each track
    df = pd.concat(features + [genres], axis=1, join='inner')

    # Save genre counts
    genre_counts = df['genre_top'].value_counts()
    genre_counts.to_csv("../data/processed/genre_counts.csv")

    # Shuffle rows to ensure randomness of row position
    df = df.sample(frac=1)
    
    return df


if __name__ == "__main__":
    # Import config file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Read audio features and track metadata information
    features = pd.read_csv("../data/raw/features.csv", index_col=0, header=[0, 1, 2])
    tracks = pd.read_csv("../data/raw/tracks.csv", index_col=0, header=[0, 1])

    # Perform feature selection
    df = select_features(features, tracks, ncomp=int(config['Preprocess']['Ncomp']))

    # Save processed dataset
    df.to_csv("../data/processed/data.csv", float_format='%g', index=False)