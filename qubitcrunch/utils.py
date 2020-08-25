import os
import sys
import inspect
import importlib
import subprocess
import numpy as np
from shutil import copyfile
from sklearn.metrics import precision_recall_fscore_support

import snorkel
from snorkel.labeling import labeling_function, LabelingFunction,LFAnalysis
from metal import LabelModel 
from snorkel.labeling import PandasLFApplier


def get_lf_performance(lf: labeling_function, X: list, y: list) -> [float, float]:
    preds = np.array([lf(x) for x in X]) # pass the lf through each example
    y = np.array(y)
    notabstain_idxes = np.where(preds != -1)[0] # get only the unabstained predictions
    if len(notabstain_idxes) == 0: # if the lf abstained on everything, metrics are pointless
        return "n/a","n/a","n/a","n/a","n/a"
    preds = preds[unabstain_idxes]
    y = y[unabstain_idxes]
    accuracy = np.mean(preds == y) # get accuracy of the unabstained predictions
    precision, recall, _, _ = precision_recall_fscore_support(y, preds)
    # LF can return both positive and negative
    if len(precision) == 2:
        class_0_precision = precision[0]
        class_0_recall = recall[0]
        class_1_precision = precision[1]
        class_1_recall = recall[1]
    else:
        # LF does not return negative
        if 1 in preds:
            class_0_precision = "n/a"
            class_0_recall = "n/a"
            class_1_precision = precision[0]
            class_1_recall = recall[0]
        # LF does not return positive
        else:
            class_0_precision = precision[0]
            class_0_recall = recall[0]
            class_1_precision = "n/a"
            class_1_recall = "n/a"

    return accuracy, class_0_precision, class_0_recall, class_1_precision, class_1_recall

def snorkel_applier(lf_list: list, in_df):
    applier = PandasLFApplier(lfs=lf_list)
    snorkel_matrix = applier.apply(df=in_df)
    #lf_analysis = LFAnalysis(snorkel_matrix, lf_list).lf_summary()
    
    return snorkel_matrix#,lf_analysis


def train_snorkel_model(snorkel_matrix, cardinality, n_epochs=500, log_freq=50, seed=42):
    snorkel_model = LabelModel(cardinality=cardinality, verbose=True)
    print("fitting snorkel model ... ")
    snorkel_model.train_model(snorkel_matrix, n_epochs=n_epochs, log_freq=log_freq, seed=seed)
    return snorkel_model


def predict_snorkel_labels(snorkel_model: LabelModel, snorkel_matrix, unlabeled_points):
    snorkel_labels = snorkel_model.predict(L=snorkel_matrix, tie_break_policy="abstain")

    pos = np.where(snorkel_labels == 1)[0]
    neg = np.where(snorkel_labels == 0)[0]
    abst = np.where(snorkel_labels == -1)[0]

    print("found " + str(len(pos)) + " positive points ... ")
    print("found " + str(len(neg)) + " negative points ... ")
    print("found " + str(len(abst)) + " abstain points ... ")

    train_queries = [unlabeled_queries[i] for i in range(len(unlabeled_queries)) if snorkel_labels[i] != -1]
    snorkel_labels = [snorkel_labels[i] for i in range(len(snorkel_labels)) if snorkel_labels[i] != -1]

    return train_queries, snorkel_labels


def get_lfs_from_module(module_name: str = 'qubitcrunch.labeling_functions', target_lf: str = ''):
    """
    Return list of labeling functions
    """
    importlib.import_module(module_name)
    all_lfs = [obj for name, obj in inspect.getmembers(sys.modules[module_name]) if isinstance(obj, snorkel.labeling.lf.core.LabelingFunction)]
    return(all_lfs)