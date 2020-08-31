import os
import sys
import inspect
import importlib
import subprocess
import numpy as np
import pandas as pd
from shutil import copyfile
from sklearn.metrics import precision_recall_fscore_support

import snorkel
from snorkel.labeling import labeling_function, LabelingFunction,LFAnalysis
from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier

import torch
import torch.optim as optim
import torch.nn as nn

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
    in_df = pd.DataFrame(in_df)
    snorkel_matrix = applier.apply(df=in_df)
    #lf_analysis = LFAnalysis(snorkel_matrix, lf_list).lf_summary()
    
    return snorkel_matrix#,lf_analysis


def train_snorkel_model(snorkel_matrix, cardinality, n_epochs=500, log_freq=50, seed=42):
    snorkel_model = LabelModel(cardinality=cardinality, verbose=True)
    print("fitting snorkel model ... ")
    snorkel_model.fit(snorkel_matrix, n_epochs=n_epochs, log_freq=log_freq, seed=seed)
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



def optimize_dual(x_tensor, w, npred_tensor, alpha_tensor, delta_tensor, lr, nepochs):
    optimizer = optim.SGD([w], lr=lr)
    
    for epoch in range(nepochs):
        ## Compute log-loss component
        A = torch.einsum('ikj,j->ik', x_tensor, w)
        B = torch.logsumexp(A, 1)
        ll = torch.sum(B)

        ## Alpha correction component
        a_corr = torch.sum(npred_tensor*alpha_tensor*w)

        ## L1 regularization
        reg = torch.sum(npred_tensor*delta_tensor*torch.abs(w))

        loss = ll - a_corr + reg

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return(w)

## label_matrix: 2d numpy array -- #(data points) x #(labelers)
## nclasses: number of classes/labels
## alphas: either 1d array of length #(labelers) or a scalar
## deltas: either 1d array of length #(labelers) or a scalar
def maxent(label_matrix, nclasses, alphas=0.9, deltas=0.1, lr=0.1, nepochs=1000):
    
    _, nlabelers = label_matrix.shape
    
    if not isinstance(alphas, np.ndarray):
        alphas = np.repeat(alphas, nlabelers)
    
    if not isinstance(deltas, np.ndarray):
        deltas = np.repeat(deltas, nlabelers)
    
    
    ## Filter for data points that have some weak labels
    inds = np.where((np.sum(label_matrix, axis=1) + nclasses) > 0 )[0]

    label_submatrix = label_matrix[inds,:]

    npoints, _ = label_submatrix.shape
    
    ## Convert to {0,1} tensor of shape #(data points) x #(classes) x #(labelers)
    one_hot_tensor = np.zeros((npoints, nclasses, nlabelers))
    npreds = np.zeros(nlabelers)
    for k in range(nclasses):
        data_inds, labeler_inds = np.where(label_submatrix == k)
        labelers, labeler_counts = np.unique(labeler_inds, return_counts=True)
        npreds[labelers] += labeler_counts
        one_hot_tensor[data_inds, np.repeat(k, len(data_inds)), labeler_inds] = 1
    
    
    ## Put everything into torch tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    x_tensor = torch.from_numpy(one_hot_tensor).float().to(device)
    npred_tensor = torch.from_numpy(npreds).float().to(device)
    alpha_tensor = torch.from_numpy(alphas).float().to(device)
    delta_tensor = torch.from_numpy(deltas).float().to(device)
    
    ## Optimize it!
    ## w = lambda from notes
    w = torch.randn(nlabelers, dtype=torch.float).to(device)
    w.requires_grad_()
    w = optimize_dual(x_tensor, w, npred_tensor, alpha_tensor, delta_tensor, lr, nepochs)
    
    ## Construct conditional probabilities
    with torch.no_grad():
        Z = torch.einsum('ikj,j->ik', x_tensor, w)
        smax = nn.Softmax(dim=1)
        P = smax(Z).cpu().numpy()
    
    ## Fill out full conditional probability matrix
    ##   Everything with no weak label gets the uniform distribution
    soft_probs = (1./nclasses)*np.ones(label_matrix.shape)
    soft_probs[inds,:] = P
    return(soft_probs)