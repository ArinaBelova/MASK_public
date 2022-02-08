from transformers import BertForTokenClassification
import torch
from transformers import BertTokenizer
import numpy as np
import nltk.data
nltk.download('punkt')

class NER_BERT(object):
    """Abstract class that other NER plugins should implement"""
    def __init__(self):
        pass

    def perform_NER(self,text):
        """Implementation of the method that should perform named entity recognition"""
        pass

    def createModel(self, text=None):
        pass

    def transform_sequences(self,tokens_labels):
        """method that transforms sequences of (token,label) into feature sequences. Returns two sequence lists for X and Y"""
        pass

    def learn(self, X_train,Y_train):
        """Function that actually train the algorithm"""
        pass

    def evaluate(self, X_test,Y_test):
        """Function to evaluate algorithm"""
        pass
