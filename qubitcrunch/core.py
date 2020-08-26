#from metal.label_model import LabelModel, EndModel
from qubitcrunch.utils import *
import random
import pandas as pd

project = {
	"data": {"labeled":[],"unlabeled":[]},
	"labeling_functions":[],
}
project["data"]["unlabeled"] = pd.read_csv("data/ag_news/ag_news_train.csv")["body"]

def batch_unlabeled_return():  # noqa: E501
    """
    Return the next batch of unlabeled documents, for now 1000 docs at random
    """
    if project["predictions"] != None:
        idx = [i for i in range(len(project["predictions"])) if .4 < project["predictions"][i] < .6]
        return list(project["data"]["unlabeled"][idx[0:999]])
    else:
        return "No batch returned"

def labeling_functions_return():  # noqa: E501
    """
    Return the list of labeling functions.
    """
    return [lf.name for lf in get_lfs_from_module()]


def weakly_label(type="snorkel"):

    if type == "random":
        predictions = [random.uniform(0, 1) for i in range(project["data"]["unlabeled"].shape[0])]
    if type == "snorkel":
        all_lfs = get_lfs_from_module()
        label_matrix = snorkel_applier(all_lfs, project["unlabeled"])
        snorkel_model = train_snorkel_model(label_matrix,project["label_cardinality"])
        predictions = predict_snorkel_labels(snorkel_model,label_matrix)
    if type == "max-ent":
        predictions = None
    project["predictions"] = predictions
