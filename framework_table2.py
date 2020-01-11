import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from Scoring import Scoring


def load_classifiers():
    clfs = []
    clfs.append({'classifier': KNeighborsClassifier(), 'name': 'k-Nearest neighbor'})
    clfs.append({'classifier': NearestCentroid(), 'name': 'Nearest mean classifier'})
    clfs.append({'classifier': DecisionTreeClassifier(), 'name': 'Decision tree'})
    clfs.append({'classifier': LogisticRegression(), 'name': 'Logistic regression'})
    clfs.append({'classifier': SVC(kernel='rbf'), 'name': 'SVM (Gaussian Kernel)'})  # also gamma='auto'?
    clfs.append({'classifier': BaggingClassifier(), 'name': 'Bagging'})
    clfs.append({'classifier': RandomForestClassifier(), 'name': 'Random Forest'})
    clfs.append({'classifier': AdaBoostClassifier(), 'name': 'AdaBoost'})
    clfs.append({'classifier': GradientBoostingClassifier(), 'name': 'Gradient Boosting Tree'})  # is this the right one?
    clfs.append({'classifier': GaussianNB(), 'name': 'Naive Bayes'})  # is this the right one?

    return clfs


def load_datasets():
    datasets = []

    data = pd.read_csv('data/csv_files/audio_files/dev_data_audio.csv')
    datasets.append({'data': data, 'modality': 'audio'})

    data = pd.read_csv('data/csv_files/text_files/dev_data_text.csv')
    datasets.append({'data': data, 'modality': 'textual'})

    data = pd.read_csv('data/csv_files/visual_files/dev_data_visual.csv')
    datasets.append({'data': data, 'modality': 'visual'})

    data = pd.read_csv('data/csv_files/visual_files/dev_data_visual_avg.csv')
    datasets.append({'data': data, 'modality': 'visual (avg)'})

    # TODO add metadata datasets

    return datasets


def load_target():
    data = pd.read_csv('data/csv_files/audio_files/dev_data_audio.csv')
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


def run_framework():
    # framework
    all_results = []
    for dataset in load_datasets():
        data = preprocess_data(dataset['data'])

        # assign features and target
        X, y = data.loc[:, data.columns != 'goodforairplane'], data.loc[:, 'goodforairplane']

        for classifier in load_classifiers():
            scoring = Scoring(classifier['name'], dataset['modality'])
            results = cross_validate(classifier['classifier'], X, y, cv=10, scoring=setup_scoring())
            scoring.precision = round(np.mean(results['test_Precision']), 3)
            scoring.recall = round(np.mean(results['test_Recall']), 3)
            scoring.f1 = round(np.mean(results['test_F1']), 3)
            all_results.append(scoring)

    return  all_results


if __name__ == "__main__":
    results = run_framework()

    list_of_tuples = []
    for i in results:
        list_of_tuples.append(i.to_tuple())

    resultsDf = pd.DataFrame(list_of_tuples, columns=['Classifier', 'Modality', 'Precision', 'Recall', 'F1'])
    print(resultsDf)
