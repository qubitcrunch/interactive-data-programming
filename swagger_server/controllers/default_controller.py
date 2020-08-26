import connexion
import six

from swagger_server import util
from qubitcrunch import core

def batch_unlabeled_return():  # noqa: E501
    """Return a batch of unlabeled data.

    GET the batch of unlabeled data. # noqa: E501


    :rtype: None
    """
    return  core.batch_unlabeled_return()


def labeling_function_new():  # noqa: E501
    """Provide a new labeling function.

    POST a new labeling function. # noqa: E501


    :rtype: None
    """
    return 'do some magic!'


def labeling_functions_return():  # noqa: E501
    """Return the list of labeling functions.

    GET the list of labeling functions # noqa: E501


    :rtype: None
    """
    return core.labeling_functions_return()


def weakly_label(weak):  # noqa: E501
    """Run algorithm to weakly label data points.

    POST an instruction for weakly labeling data points # noqa: E501

    :param weak: name of weak labeling algorithm to be used
    :type weak: str

    :rtype: None
    """
    return 'do some magic!'
