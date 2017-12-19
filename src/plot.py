import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


def plot_pca(data, labels, output):
    pca = PCA(n_components=2)
    pca_df = pca.fit_transform(data)

    xs = pca_df[:, 0]
    ys = pca_df[:, 1]

    plt.figure()
    plt.scatter(xs, ys, c=labels, s=10, alpha=0.3)
    plt.title("Plotted PCA features")
    plt.xlabel("PCA Feature #1")
    plt.ylabel("PCA Feature #2")
    
    plt.savefig(output, bbox_inches='tight', dpi=150)


def plot_forest_ns(filename, output):
    ns = pd.read_csv(filename)

    plt.figure()
    plt.plot(ns['N'], ns['Mean_accuracy'])
    plt.title("Number of trees vs cross-validation mean accuracy")
    plt.xticks(ns['N'], ns['N'], rotation=45)
    plt.xlabel("Number of trees")
    plt.ylabel("CV mean accuracy")
    
    plt.savefig(output, bbox_inches='tight', dpi=150)


if __name__ == "__main__":
    df = pd.read_csv("../data/processed/data.csv")

    # Convert genre names to discrete integer value for training
    df['genre_top'] = LabelEncoder().fit_transform(df['genre_top'])

    rows = np.random.choice(df.index.values, int(len(df) * 0.1), replace=False)
    df = df.ix[rows]

    X = df.drop(['genre_top'], axis=1)
    y = df['genre_top']

    plot_pca(X, y, "../images/plot_pca.png")
    plot_forest_ns("../data/processed/forest_ns.csv", "../images/plot_forest.png")