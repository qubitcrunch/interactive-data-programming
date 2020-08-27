import connexion
import six

from swagger_server import util
import sys
from qubitcrunch import core
import importlib
import inspect
import snorkel

def batch_unlabeled_return():  # noqa: E501
    """Return a batch of unlabeled data.

    GET the batch of unlabeled data. # noqa: E501


    :rtype: None
    """
    return  core.batch_unlabeled_return()


def labeling_function_new(labeling_function_str):  # noqa: E501
    """Provide a new labeling function.

    POST a new labeling function. # noqa: E501


    :rtype: None
    """
    core.labeling_function_new(labeling_function_str)


def labeling_functions_return():  # noqa: E501
    """Return the list of labeling functions.

    GET the list of labeling functions # noqa: E501


    :rtype: None
    """
    importlib.import_module("qubitcrunch.labeling_functions")
    all_lfs = [obj for name, obj in inspect.getmembers(sys.modules["qubitcrunch.labeling_functions"]) if isinstance(obj, snorkel.labeling.lf.core.LabelingFunction)]
    return [lf.name for lf in all_lfs]


def weakly_label(weak):  # noqa: E501
    """Run algorithm to weakly label data points.

    POST an instruction for weakly labeling data points # noqa: E501

    :param weak: name of weak labeling algorithm to be used
    :type weak: str

    :rtype: None
    """
    return core.weakly_label(weak)
