"""
Copyright 2020 ICES, University of Manchester, Evenset Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
"""
    *evaluate.py* - Evaluate algorithm on test data from i2b2 
    IMPORTANT: you need to have i2b2 data in order to evaluate, as algoithm works for now only with the i2b2 layout data
    Example of starting: python evaluate.py --source_location "../testing-PHI-Gold-fixed/" --algorithm NER_BERT
    Code by: Arina Belova
"""

import argparse
import importlib

from utils.readers import read_i2b2_data
import utils.spec_tokenizers

if __name__ == "__main__":

    """
    Evaluates algorithm of selection on test data
    """

    print("Evaluating framework")
    parser = argparse.ArgumentParser(description='Evaluation framework for Named Entity recognition')
    parser.add_argument('--source_location', help='source location of the dataset on your hard disk')
    parser.add_argument('--algorithm', help='algorithm to use')
    #parser.add_argument('--algorithm_location', help='location of the algorithm')
    args = parser.parse_args()
    path_to_data = args.source_location
   # path_to_alg = args.algorithm_location
    documents = read_i2b2_data(path_to_data)
    if documents== None:
        print("Error: No input source is defined")
        exit(2)

    tokens_labels = utils.spec_tokenizers.tokenize_to_seq(documents)
    package = "ner_plugins."+ args.algorithm
    algorithm = args.algorithm
    inpor = importlib.import_module(package)
    # find a class and instantiate
    class_ = getattr(inpor, algorithm)
    instance = class_()

    X,Y = instance.transform_sequences(tokens_labels)
    instance.evaluate(X,Y)
   
    print("Done!")
