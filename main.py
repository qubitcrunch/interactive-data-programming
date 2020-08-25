import torch
import torchtext
from torchtext.datasets import text_classification
import metal
import scipy as sp
NGRAMS = 2
import os
if not os.path.isdir('./.data'):
    os.mkdir('./.data')
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root='./.data', ngrams=NGRAMS, vocab=None)
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import IPython
IPython.embed()
"""
data = 
n = # data points
m = # labeling functions
k = cardinality of the classification task

Load for each split:
L: an [n,m] scipy.sparse label matrix of noisy labels
Y: an n-dim numpy.ndarray of target labels
X: an n-dim iterable (e.g., a list) of end model inputs


from metal.label_model import LabelModel, EndModel

# Train a label model and generate training labels
label_model = LabelModel(k)
label_model.train_model(L_train)
Y_train_probs = label_model.predict_proba(L_train)

# Train a discriminative end model with the generated labels
end_model = EndModel([1000,10,2])
end_model.train_model(train_data=(X_train, Y_train_probs), valid_data=(X_dev, Y_dev))

# Evaluate performance
score = end_model.score(data=(X_test, Y_test), metric="accuracy")
"""

