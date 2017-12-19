import sys
import configparser
import numpy as np
import pandas as pd
from collections import Counter

from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.preprocessing import LabelEncoder


def test_n(X, y, start, end, step):
    """
    Test all possible n_estimator values for Random Forest

    :param X: Dataset of all features
    :param y: Labels for dataset
    :param start: N value to start at
    :param end: N value to end at
    :returns: DataFrame of all N values and their corresponding accuracy
    """
    arr = []
    
    # Test N vals from start -> end with step=10
    for n in range(start, end+1, 25):
        # Perform cross-validation on N value and return mean accuracy
        print("Iteration for N=%d" % n)
        clf = XGBClassifier(n_estimators=n, max_depth=10)
        scores = cross_validate(clf, X, y, cv=int(config_train['Split']), scoring='accuracy')
        mean = np.mean(scores['test_score'])
        arr.append([n, mean])
        
    return pd.DataFrame(arr, columns=['N', 'Mean_accuracy'])


def validate_model(clf, X, y, cv):
    """
    Validate the model using cross-validation and output CV accuracy and F1 score

    :param clf: Classifier being trained
    :param X: Dataset of all features
    :param y: Labels for dataset
    :param cv: Cross-validation split
    """
    # Cross-validate and retrieve scores for accuracy and F1
    scores = cross_validate(clf, X, y, cv=cv, scoring=['accuracy', 'f1_weighted'])

    # Output the individual scores and their means
    str_accuracy = ' '.join([format(s, ".2f") for s in scores['test_accuracy']])
    str_f1 = ' '.join([format(s, ".2f") for s in scores['test_f1_weighted']])        
    print("Accuracy = %s; Mean = %.2f" % (str_accuracy, np.mean(scores['test_accuracy'])))
    print("Weighted F1 Score = %s; Mean = %.2f" % (str_f1, np.mean(scores['test_f1_weighted'])))


def common_predictions(clf, X, y, encoder, cv):
    """
    Find common incorrect predictions to see what the model mistakes certain genres for

    :param clf: Classifier being trained
    :param X: Dataset of all features
    :param y: Labels for dataset
    :param encoder: LabelEncoder to transform predictions into genre names
    :param cv: Cross-validation split
    """
    # Retrieve predictions based on cross-validation
    predictions = cross_val_predict(clf, X, y, cv=cv)

    # Retrieve genres names based on numerical column value
    pairs = zip(encoder.inverse_transform(predictions), encoder.inverse_transform(y.as_matrix()))
    pairs = dict(Counter(pairs))

    # Select most common incorrect predictions
    pairs = sorted(pairs, reverse=True, key=pairs.get)
    pairs = pairs[:int(len(pairs)*0.2)]
    pairs = list({tuple(sorted(k)) for k in pairs if k[0] != k[1]})

    print(pairs)


def write_predictions(filename, clf, X, y, encoder, cv):
    predictions = cross_val_predict(clf, X, y, cv=cv)
    decoded_predict = encoder.inverse_transform(predictions)

    unique, counts = np.unique(decoded_predict, return_counts=True)
    matrix = np.column_stack((unique, counts))

    df = pd.DataFrame(matrix)
    df.to_csv(filename, header=False, index=False)


if __name__ == "__main__":
    print("Reading config_train file and CSV...")
    config = configparser.ConfigParser()
    config.read('config.ini')
    config_train = config['Train']
    config_model = config['Model']

    df = pd.read_csv("../data/processed/data.csv")
    print("Done.")
    
    # Set parameters and instantiate XGB classifier instance
    params = {
        "n_estimators": int(config_model['NumTrees']),
        "max_depth": int(config_model['MaxDepth'])
        }
    clf = XGBClassifier(**params)

    # Convert genre names to discrete integer value for training
    encoder = LabelEncoder()
    df['genre_top'] = encoder.fit_transform(df['genre_top'])
    
    # Split X and y from dataset
    X = df.drop(['genre_top'], axis=1)
    y = df['genre_top']

    # Perform cross-validation
    if config_train['Mode'].lower() == 'testmodel':
        print("Performing XGBoost cross-validation...")
        validate_model(clf, X, y, cv=int(config_train['Split']))

    # Find common incorrect predictions
    elif config_train['Mode'].lower() == 'findcommon':
        print("Finding common XGBoost cross-validation incorrect predictions...")
        common_predictions(clf, X, y, encoder, cv=int(config_train['Split']))

    elif config_train['Mode'].lower() == 'predict':
        print("Predicting each label through cross-validation...")
        write_predictions("../data/processed/predictions.csv", clf, X, y, encoder, cv=int(config_train['Split']))

    # Calculate accuracy for n_estimators in random forest classifier
    elif config_train['Mode'].lower() == 'testn':
        print("Calculating mean accuracy for different number of trees...")
        ns = test_n(X, y, int(config_train['StartN']), int(config_train['EndN']), int(config_train['StepN']))
        ns.to_csv("../data/processed/forest_ns.csv", float_format='%g', index=False)

    else:
        print("Please specify a valid mode in config")