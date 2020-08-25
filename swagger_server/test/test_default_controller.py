# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.test import BaseTestCase


class TestDefaultController(BaseTestCase):
    """DefaultController integration test stubs"""

    def test_batch_unlabeled_return(self):
        """Test case for batch_unlabeled_return

        Return a batch of unlabeled data.
        """
        response = self.client.open(
            '/qubitcrunch/interactive-data-programming/1.0.0/batch_unlabeled',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_labeling_function_new(self):
        """Test case for labeling_function_new

        Provide a new labeling function.
        """
        response = self.client.open(
            '/qubitcrunch/interactive-data-programming/1.0.0/labeling_function_new',
            method='POST')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_labeling_functions_return(self):
        """Test case for labeling_functions_return

        Return the list of labeling functions.
        """
        response = self.client.open(
            '/qubitcrunch/interactive-data-programming/1.0.0/labeling_functions',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_weakly_label(self):
        """Test case for weakly_label

        Run algorithm to weakly label data points.
        """
        response = self.client.open(
            '/qubitcrunch/interactive-data-programming/1.0.0/weakly_label',
            method='POST')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
