from qubitcrunch.utils import *
import pandas as pd
from scipy.stats import entropy

project = {
	"data": {"labeled":[],"unlabeled":[],"predictions":[]},
	"labeling_functions":[],
}
project["data"]["unlabeled"] = pd.read_csv("data/ag_news/ag_news_train.csv")["body"]
project["data"]["label_cardinality"] = 4

def batch_unlabeled_return():  # noqa: E501
    """
    Return the next batch of unlabeled documents, for now 1000 docs at random
    """

    entropies = [entropy(row,base=2) for row in project["data"]["predictions"]]
    sorted = np.argsort(entropies)
    idx = sorted[::-1][:1000]

    return list(project["data"]["unlabeled"][idx])

def labeling_functions_return():  # noqa: E501
    """
    Return the list of labeling functions.
    """

    return [lf.name for lf in get_lfs_from_module()]

def labeling_function_new(labeling_function_str,module_file_path="qubitcrunch/labeling_functions.py",):

    with open(module_file_path,"a") as file:
        file.write("\n\n")
        file.write("@labeling_function()\n")
        file.write(labeling_function_str)
    file.close()

def weakly_label(weak="snorkel"):

    if weak == "snorkel":
        all_lfs = get_lfs_from_module()
        label_matrix = snorkel_applier(all_lfs, project["data"]["unlabeled"])
        snorkel_model = train_snorkel_model(label_matrix, cardinality=project["data"]["label_cardinality"])
        snorkel_labels = snorkel_model.predict_proba(L=label_matrix)
        project["data"]["predictions"] = snorkel_labels