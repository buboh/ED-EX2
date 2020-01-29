import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from Scoring import Scoring
import random
import warnings
warnings.filterwarnings("ignore")


def load_classifiers():
    clfs = []
    clfs.append({'classifier': KNeighborsClassifier(), 'name': 'k-Nearest neighbor'})
    clfs.append({'classifier': NearestCentroid(), 'name': 'Nearest mean classifier'})
    clfs.append({'classifier': DecisionTreeClassifier(), 'name': 'Decision tree'})
    clfs.append({'classifier': LogisticRegression(), 'name': 'Logistic regression'})
    clfs.append({'classifier': SVC(kernel='rbf'), 'name': 'SVM (Gaussian Kernel)'})
    clfs.append({'classifier': BaggingClassifier(), 'name': 'Bagging'})
    clfs.append({'classifier': RandomForestClassifier(), 'name': 'Random Forest'})
    clfs.append({'classifier': AdaBoostClassifier(), 'name': 'AdaBoost'})
    clfs.append({'classifier': GradientBoostingClassifier(), 'name': 'Gradient Boosting Tree'})
    clfs.append({'classifier': GaussianNB(), 'name': 'Naive Bayes'})

    return clfs


def load_datasets():
    datasets = []

    data = pd.read_csv('data/csv_files/audio_files/dev_data_audio.csv')
    datasets.append({'data': data, 'modality': 'audio'})

    data = pd.read_csv('data/csv_files/text_files/dev_data_text.csv')
    datasets.append({'data': data, 'modality': 'textual'})

    data = pd.read_csv('data/csv_files/visual_files/dev_data_visual_avg.csv')
    datasets.append({'data': data, 'modality': 'visual'})

    data = pd.read_csv('data/csv_files/metadata_files/dev_data_meta.csv')
    datasets.append({'data': data, 'modality': 'metadata'})

    return datasets

def load_test_dataset(modality, features_position):
    if modality == 'audio':
        data = pd.read_csv('data/csv_files/audio_files/test_data_audio.csv')
    if modality == 'textual':
        data = pd.read_csv('data/csv_files/text_files/test_data_text.csv')
    if modality == 'visual':
        data = pd.read_csv('data/csv_files/visual_files/test_data_visual_avg.csv')
    if modality == 'metadata':
        data = pd.read_csv('data/csv_files/metadata_files/test_data_meta.csv')

    data = preprocess_data(data)
    result = data.iloc[:,features_position]
    return result

def load_target():
    data = pd.read_csv('data/csv_files/audio_files/dev_data_audio.csv')
    return data['goodforairplane']

def load_test_target():
    data = pd.read_csv('data/csv_files/audio_files/test_data_audio.csv')
    return data['goodforairplane']


def preprocess_data(data):
    df = data
    if 'movie' in df.columns:
        df.drop(columns=['movie'], inplace=True)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    if not 'goodforairplane' in df.columns:
        df['goodforairplane'] = load_target()

    return df


def setup_scoring():
    return {
        'Precision': make_scorer(precision_score),
        'Recall': make_scorer(recall_score),
        'F1': make_scorer(f1_score),
    }

def lvw(X, y, iteration_number, initial_f1, classifier):
    best_f1 = initial_f1
    feature_space = list(set(range(X.shape[1])))
    best_features = feature_space
    iteration = 0
    while iteration < iteration_number:
        selected_features = list(set(random.sample(feature_space, random.randint(1, len(feature_space)))))
        selected_subspace = X.iloc[:,selected_features]
        results = cross_validate(classifier, selected_subspace, y, cv=10, scoring='f1')
        avg_result = np.mean(results['test_score'])
        if avg_result > best_f1 or (avg_result == best_f1 and len(selected_features) < len(feature_space)):
            iteration = 0
            best_features = selected_features
            best_f1 = avg_result
        else:
            iteration += 1

        return best_f1, best_features

def run_table2():
    # framework
    all_results = []
    selected_classifiers = []
    for dataset in load_datasets():
        data = preprocess_data(dataset['data'])

        # assign features and target
        X, y = data.loc[:, data.columns != 'goodforairplane'], data.loc[:, 'goodforairplane']

        for classifier in load_classifiers():
            scoring = Scoring(classifier['name'], dataset['modality'])

            # get scores with all features
            results = cross_validate(classifier['classifier'], X, y, cv=10, scoring=setup_scoring())
            scoring.precision = round(np.mean(results['test_Precision']), 3)
            scoring.recall = round(np.mean(results['test_Recall']), 3)
            scoring.f1 = round(np.mean(results['test_F1']), 3)

            # run las vegas wrapper
            best_f1, best_features = lvw(X, y, 100, scoring.f1, classifier['classifier'])
            selected_subspace = X.iloc[:, best_features]

            # run with last vegas selected features
            results = cross_validate(classifier['classifier'], selected_subspace, y, cv=10, scoring=setup_scoring())
            scoring.precision = round(np.mean(results['test_Precision']), 3)
            scoring.recall = round(np.mean(results['test_Recall']), 3)
            scoring.f1 = round(np.mean(results['test_F1']), 3)
            scoring.best_features = len(best_features)

            if scoring.precision > 0.5 and scoring.recall > 0.5 and scoring.f1 > 0.5:
                all_results.append(scoring)

                # storing selected classifiers for stacking
                selected_classifiers.append(
                    {'classifier': classifier['classifier'],
                     'modality': dataset['modality'],
                     'subspace': selected_subspace,
                     'features_position': best_features,
                     'test': y,
                     })

    return all_results, selected_classifiers


def fit_multiple_estimators(classifiers, X_list, y_list):
    return [clf.fit(X, y) for clf, X, y in zip(classifiers, X_list, y_list)]

def cv_estimator(estimators, X_list, y):
    predictions = np.asarray([cross_val_predict(clf, X, y, cv=10) for clf, X in zip(estimators, X_list)]).T
    return predictions

def test_estimator(estimators, X_test_list):
    predictions = np.asarray([clf.predict(X_test) for clf, X_test in zip(estimators, X_test_list)]).T
    return predictions

def majority_voting(predictions, weights=None):
    majority = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=weights)),
                                   axis=1,
                                   arr=predictions.astype('int'))
    return majority

def run_table3(selected_classifiers):
    feature_subspaces = []
    classifiers = []
    targets = []
    X_test_list = []
    for i in selected_classifiers:
        classifiers.append(i['classifier'])
        feature_subspaces.append(i['subspace'])
        targets.append(i['test'])
        X_test_list.append(load_test_dataset(i['modality'], i['features_position']))
    fitted_estimators = fit_multiple_estimators(classifiers, feature_subspaces, targets)

    # Majority voting (test)
    prediction = majority_voting(test_estimator(fitted_estimators, X_test_list))
    print('Voting (test)')
    print('Precision: ' + str(precision_score(load_test_target(), prediction)))
    print('Recall: ' + str(recall_score(load_test_target(), prediction)))
    print('F1: ' + str(f1_score(load_test_target(), prediction)))

    # Majority voting (cv)
    prediction = majority_voting(cv_estimator(fitted_estimators, feature_subspaces, load_target()))
    print('Voting (cv)')
    print('Precision: ' + str(precision_score(load_target(), prediction)))
    print('Recall: ' + str(recall_score(load_target(), prediction)))
    print('F1: ' + str(f1_score(load_target(), prediction)))

    # Label stacking (cv)
    predictions = cv_estimator(fitted_estimators, feature_subspaces, load_target())
    predictions_df = pd.DataFrame(data=predictions)
    results = cross_validate(LogisticRegression(), predictions_df, load_target(), cv=10, scoring=setup_scoring())
    print('Label stacking (cv)')
    print('Precision: ' + str(round(np.mean(results['test_Precision']), 3)))
    print('Recall: ' + str(round(np.mean(results['test_Recall']), 3)))
    print('F1: ' + str(round(np.mean(results['test_F1']), 3)))

    # Label stacking (test)
    predictions = test_estimator(fitted_estimators, X_test_list)
    predictions_df = pd.DataFrame(data=predictions)
    results = cross_validate(LogisticRegression(), predictions_df, load_test_target(), cv=10, scoring=setup_scoring())
    print('Label stacking (test)')
    print('Precision: ' + str(round(np.mean(results['test_Precision']), 3)))
    print('Recall: ' + str(round(np.mean(results['test_Recall']), 3)))
    print('F1: ' + str(round(np.mean(results['test_F1']), 3)))


if __name__ == "__main__":
    results_lvw, selected_classifiers = run_table2()

    list_of_tuples = []
    for i in results_lvw:
        list_of_tuples.append(i.to_tuple())

    results_lvwDf = pd.DataFrame(list_of_tuples, columns=['Classifier', 'Modality', 'Precision', 'Recall', 'F1', 'Best Features'])
    results_lvwDf.to_csv('scikit0.22_lvw100.csv')

    run_table3(selected_classifiers)

