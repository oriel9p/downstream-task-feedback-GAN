
import pandas as pd
import numpy as np
from copy import copy
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score, roc_curve, auc, precision_recall_curve, f1_score
import warnings
import torch
warnings.filterwarnings("ignore")

# from pandas.core.common import SettingWithCopyWarning
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from sklearn import preprocessing
import matplotlib.pyplot as plt

# ---------- FUNCTIONS -------------- #


# Split data to train and test - v0.2
def data_split(cohort, train_cols, target_col, test_size, stratifier=None):
    X, y = cohort[train_cols], cohort[target_col]
    X_train, X_test, y_train, y_test = train_test_split \
        (X, y, test_size=test_size, stratify=stratifier, random_state=42)
    return X_train, X_test, y_train, y_test


# train prediction model
def fit_model(X_train, y_train, model='xgboost'):
    # Model object creation and fit
    if model == 'xgboost':
        model_fitted = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                  max_depth=8, random_state=42).fit(X_train, y_train)
    else:
        raise ValueError('Model type not in list')
    return (model_fitted)


# model evaluation
def model_evaluate(model, X_test, y_test, eval_criteria='roc_auc'):
    e_metric = 0
    if eval_criteria == 'roc_auc':
        e_metric = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    elif eval_criteria == 'precision':
        e_metric = precision_score(y_test, model.predict(X_test))
    elif eval_criteria == 'accuracy':
        e_metric = accuracy_score(y_test, model.predict(X_test))
    elif eval_criteria == 'recall':
        e_metric = recall_score(y_test, model.predict(X_test))
    elif eval_criteria == 'pr_auc':
        y_score = model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)
        e_metric = auc(recall, precision)
    elif eval_criteria == 'f1':
        e_metric = f1_score(y_test, model.predict(X_test))
    else:
        raise ValueError('Evaluation criteria not in list')
    return e_metric


def train_generator(cohort,metadata, discrete_columns, epochs, batch_size, X_test, y_test, label, params=None):
    # original usage
    # generator_model = CTGANSynthesizer(epochs=300, batch_size=250)
    generator_model = CTGANSynthesizer(
         epochs=epochs, batch_size=batch_size,verbose=True, metadata=metadata, X_test=X_test, y_test=y_test, label=label
    )
    generator_model.fit(cohort)
    return generator_model


def generate_samples(cohort, generator_model, n_samples):
    # Synthetic copy generation
    num_of_samples = int(n_samples / 100 * cohort.shape[0])
    print("Generating " + str(num_of_samples))
    samples = generator_model.sample(num_of_samples)
    return samples

def import_n_setup(use_case):
    if use_case=='heart':
        data = pd.read_csv('datasets/heart_fail.csv')
        discrete_columns = ['Sex','ChestPainType','RestingECG',
                            'ExerciseAngina','ST_Slope','HeartDisease']
        label = 'HeartDisease'
    elif use_case=='adult':
        data = pd.read_csv('datasets/adults_concesus_income.csv')
        discrete_columns = ['workclass','education','marital-status',
                            'occupation','relationship','race','sex','native-country']
        label = 'target'
    elif use_case=='students':
        data = pd.read_csv('datasets/students_dropout.csv')
        discrete_columns = ['Marital status', 'Application order', 'Attendance',
                            'Previous qualification', 'Mom Qualification', 'Dad Qualification',
                            'Mom Occupation', 'Dad Occupation', 'Educational special needs','Debtor','Gender',
                            'Scholarship holder']
        label = 'Target'
    else:
        ValueError('No such dataset available')
        return None

    return data, discrete_columns, label
if __name__ == '__main__':
    # check for GPU
    print(torch.cuda.is_available())
    results_dict = {'dataset': [], 'n': [], 'roc_auc': [], 'syn_roc_auc': [], 'f1': [],
                    'syn_f1': []}  # 'precision': [], 'recall': [], 'accuracy': []}

    ucs = ['heart']
    for use_case in ucs:
        data, discrete_columns, label = import_n_setup(use_case)
        results_dict['dataset'].append(use_case)
        results_dict['n'].append(data.shape[0])
        # Train an XGBOOST
        X, y = data.loc[:, data.columns != label], data[label]
        X_train, X_test, y_train, y_test = train_test_split \
            (X, y, test_size=0.2, stratify=data[label], random_state=42)

        xg_model = fit_model(X_train, y_train)
        roc_auc = model_evaluate(xg_model,X_test, y_test, eval_criteria='roc_auc')
        f1 = model_evaluate(xg_model, X_test, y_test, eval_criteria='f1')
        results_dict['roc_auc'].append(roc_auc)
        results_dict['f1'].append(f1)

        # generate synthetic data
        train_data = X_train
        train_data[label] = y_train
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=data)
        ctgan_model = train_generator(train_data,metadata,discrete_columns, 10, 100, X_test, y_test, label) # TODO: added, X_test set, y_test set and label column name
        samples = ctgan_model.sample(1000) # generate equal size synthetic


        # train on synthetic data
        sX, sy = samples.loc[:, samples.columns != label], samples[label]

        xg_model = fit_model(sX, sy)
        syn_roc_auc = model_evaluate(xg_model, X_test, y_test, eval_criteria='roc_auc')
        syn_f1 = model_evaluate(xg_model, X_test, y_test, eval_criteria='f1')
        results_dict['syn_roc_auc'].append(syn_roc_auc)
        results_dict['syn_f1'].append(syn_f1)
        df = pd.DataFrame(results_dict)
    df.to_csv('results.csv')


