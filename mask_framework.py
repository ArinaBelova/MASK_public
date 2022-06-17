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

"""*mask_framework.py* --
Main MASK Framework module
Code by: Nikola Milosevic
"""
from os import listdir, path, mkdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import importlib

import datetime
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize.util import align_tokens
import re

# To calculate execution time
import time

# TODO: some algorithms like CRF and its variations do not like files to mask in .xml format, which causes pai for the evaluation on test set of their performance.
# I am not sure why this problem was not addressed earlier, but that is for the future (summer?) to solve.

# CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

# # Clean xml file from xml tags not to crash de-identification algorithms (except BERT, BERT is ok with it)
# def cleanxml(raw_xml_content):
#   cleantext = re.sub(CLEANR, '', raw_xml_content)
#   return cleantext

class Configuration():
    """Class for reading configuration file
        Init function that can take configuration file, or it uses default location:
        configuration.cnf file in folder where mask_framework is
    """
    def __init__(self, configuration="configuration.cnf"):
        """Init function that can take configuration file, or it uses default location:
            configuration.cnf file in folder where mask_framework is
        """
        self.conf = configuration
        conf_doc = ET.parse(self.conf)
        root = conf_doc.getroot()
        #print(root.text)
        self.entities_list = []
        for elem in root:
            if elem.tag == "project_name":
                self.project_name = elem.text
            if elem.tag == "project_start_date":
                self.project_start_date = elem.text
            if elem.tag == "project_owner":
                self.project_owner = elem.text
            if elem.tag == "project_owner_contact":
                self.project_owner_contact = elem.text
            if elem.tag == "algorithms":
                for entities in elem:
                    entity = {}
                    for ent in entities:
                        if ent.tag == "algorithm":
                            # if entity["algorithm"] has subelements <algorithm1>, <algorithm2>
                            if list(ent):
                                list_alg = []
                                for algorithm in ent:
                                    list_alg.append(algorithm.text)
                                entity[ent.tag] = list_alg
                            # if there is only one algorithm and no nested structure in entity["algorithm"]
                            else:
                                entity[ent.tag] = [ent.text]
                        # for the rest of tags in <entity>
                        else:
                            entity[ent.tag] = ent.text
                    self.entities_list.append(entity)

            # For several algorithms on one entity we may apply either union of labels idea or intersection of labels idea
            # TODO: describe more in detail what each option means.
            if elem.tag == "resolution":
                self.resolution = elem.text

            if elem.tag == "dataset":
                for ent in elem:
                    if ent.tag == "dataset_location":
                        self.dataset_location = ent.text
                    if ent.tag == "data_output":
                        self.data_output = ent.text

_treebank_word_tokenizer = TreebankWordTokenizer()

def consolidate_NER_results(final_sequences, text):
    """
    Function that from a list of sequences returned from the NER function is updated with spans
    :param final_sequences: Sequences returned from NER function. Sequence is a array of arrays of tokens in format (token,label).
    :param text: full text article
    :return: a list of tuples that includes spans in the following format: (token,label,span_begin,span_end)
    """
    tokens = []
    # Get all the tokens from final_sequences, which consist of pairs (token, label)
    for a in final_sequences:
        for b in a:
            tokens.append(b[0])

    spans = align_tokens(tokens, text) # get the list of spans for these tokens
    fin = []
    multiplier = 0
    for i in range(0, len(final_sequences)):
        #multiplier = 0
        if i > 0:
            multiplier = multiplier + len(final_sequences[i-1]) # a hack by Nikola: pointer on the start of the token
            #subtractor = 1
        for j in range(0, len(final_sequences[i])):
            token = final_sequences[i][j][0]
            label = final_sequences[i][j][1]
            span_min = spans[multiplier+j][0]
            span_max = spans[multiplier+j][1]
            # Maybe not the most elegant soluion to BERT tokenisation issue, when punctuation
            # signs are considered to be labels.
            if (token == "/" or token == "," or token == "-" or token == "\\" or token == ".") and (label == "DATE" or label == "NAME" or label == "LOCATION" or label == "ID" or label == "CONTACT" or label == "PROFESSION"):
                label = "O"
            fin.append((token, label, span_min, span_max))
    return fin

def recalculate_tokens(token_array, index, token_size, replacement_size, new_text, new_token):
    """
    Function that recalculates token spans when the token is replaced

    :param token_array: Array of tokens with all information, including label and spans
    :param index: Index of the token in the array that is being replaced
    :param token_size: size of the token that is being replaced
    :param replacement_size: size of the new token that is replacing token
    :param new_text: whole text (have been used for debugging purposes, not obsolete and can be empty string)
    :param new_token: New string that is replacing the token.
    :return: new, modified list of tokens with information about labels and spans. Basically list of tuples (token,label,start_span,end_span)
    """
    shift = replacement_size - token_size
    #print(f"Our token array of work is {token_array}")
    #print(f"Shift for token at index {index} is {shift}. Will replace this token with new token {new_token} of size {replacement_size}")
    new_token_array = []
    for i in range(0, len(token_array)):
        if i == index:
            new_start = token_array[i][2] #+ shift
            new_end = token_array[i][3] + shift
            new_token_array.append((new_token, token_array[i][1], new_start, new_end))
        elif i > index:
            new_start = token_array[i][2] + shift
            new_end = token_array[i][3] + shift
            new_token_array.append((token_array[i][0], token_array[i][1], new_start, new_end))
        else:
            new_token_array.append(token_array[i])

    return new_token_array


def compare_results(resolution, entity_name, alg_result_1, alg_result_2, mask_running_log):
    # Now the fun part
    # We consider 3-4 cases:

    overall_result = []

    # loop over indices of arrays as we will need to look in thee future as well as in the past
    for alg1_idx, alg2_idx in zip(range(len(alg_result_1)), range(len(alg_result_2))): # alg1 and alg2 of shape: (token, label, span_min, span_max)
        # Assert that we compare elements with the same tokens

        assert alg_result_1[alg1_idx][0] == alg_result_2[alg2_idx][0], f"Tokens {alg_result_1[alg1_idx][0]} and {alg_result_2[alg2_idx][0]} are not equal"
        # If labels or spans are not equal, this is bad. You need to align outputs of models in the fashin of BERT.
        # To be discussed in the documentation.

        assert alg_result_1[alg1_idx][2] == alg_result_2[alg2_idx][2], "Mismatch of span (lower bound) for a pair of comparable tokens!"
        assert alg_result_1[alg1_idx][3] == alg_result_2[alg2_idx][3], "Mismatch of span (upper bound) for a pair of comparable tokens!"

        # print(f"My tag for algo1 is {alg_result_1[alg1_idx][1]}")
        # print(f"My tag for algo2 is {alg_result_2[alg2_idx][1]}")
        # TODO: solve alignment problems to the style of BERT alignment. Need to change al CRF algorithms and check fro Glove BiLSTM.
        # CASE 1: If for certain token algorithm1 returns "O" and algorithm2 returns "ENTITY_NAME", use "ENTITY_NAME" overal

        alg1_pred = alg_result_1[alg1_idx][1]
        alg2_pred = alg_result_2[alg2_idx][1]

        result = []

        if resolution == "union":
            union = ""
            if (alg1_pred == "O") and (alg2_pred != "O"):
                mask_running_log.write(f"Algorithm 1 recognised token {alg_result_1[alg1_idx][0]} with span ({alg_result_1[alg1_idx][2]},{alg_result_1[alg1_idx][3]}) as 'O' and algorithm 2 recognised it as {alg2_pred}" + "\n")
                union = alg2_pred
            elif (alg2_pred == "O") and (alg1_pred != "O") and (alg1_pred != "PAD"):
                mask_running_log.write(f"Algorithm 1 recognised token {alg_result_1[alg1_idx][0]} with span ({alg_result_1[alg1_idx][2]},{alg_result_1[alg1_idx][3]}) as {alg1_pred} and algorithm 2 recognised it as 'O'"+ "\n")
                union = alg1_pred
            elif (alg1_pred != "O") and (alg2_pred != "O"):
                if alg1_pred == alg2_pred:
                    union = alg1_pred
                # it may happen, in that case take a token recognised with current entity_name we cater for (?)
                elif (alg1_pred == entity_name) or (alg2_pred == entity_name):
                    token = alg_result_1[alg1_idx][0]
                    low = alg_result_1[alg1_idx][2]
                    high = alg_result_1[alg1_idx][3]
                    # if labels are not equal -> choose label of leading algorithm 1
                    mask_running_log.write(f"There is a debate: Algorithm 1 recognised a token {token} with span ({low}, {high}) as {alg1_pred} and Algorithm 2 recognised it as {alg2_pred}. Right now we classify for entiy {entity_name}"
                    + "\n" + f"Human analyst should look at this problem manually, for now this entity is set as {alg1_pred} as algorithm 1 is the lead"
                    + "\n" + "----------------------------------------------------------------------------------------------------------------------------------------------------------------------"+ "\n")
                    union = alg1_pred
                    #raise Exception(f"Problem of intersection of dstinct token labels has occured alg1 recognised it as {alg1_pred} and alg2 recognised it as {alg2_pred}")
                # else:
                #     union = "O"
            else:
                union = "O"
            result.append(alg_result_1[alg1_idx][0])    # token
            result.append(union) # label
            result.append(alg_result_1[alg1_idx][2]) # lower bound of span
            result.append(alg_result_1[alg1_idx][3]) # upper bound of span
            result = tuple(result)

        elif resolution == "intersection":
            intersection = ""
            if (alg1_pred == "O") and (alg2_pred != "O"):
                mask_running_log.write(f"Algorithm 1 recognised token {alg_result_1[alg1_idx][0]} with span ({alg_result_1[alg1_idx][2]},{alg_result_1[alg1_idx][3]}) as 'O' and algorithm 2 recognised it as {alg2_pred}" + "\n")
                intersection = "O"
            elif (alg2_pred == "O") and (alg1_pred != "O") and (alg1_pred != "PAD"):
                mask_running_log.write(f"Algorithm 1 recognised token {alg_result_1[alg1_idx][0]} with span ({alg_result_1[alg1_idx][2]},{alg_result_1[alg1_idx][3]}) as {alg1_pred} and algorithm 2 recognised it as 'O'"+ "\n")
                intersection = "O"
            elif (alg1_pred != "O") and (alg2_pred != "O") and (alg1_pred == alg2_pred):
                    intersection == alg1_pred
            elif (alg1_pred != "O") and (alg2_pred != "O") and (alg1_pred != alg2_pred) and ((alg1_pred == entity_name) or (alg2_pred == entity_name)):
                token = alg_result_1[alg1_idx][0]
                low = alg_result_1[alg1_idx][2]
                high = alg_result_1[alg1_idx][3]
                # if labels are not equal -> choose label of leading algorithm 1
                mask_running_log.write(f"There is a debate: Algorithm 1 recognised a token {token} with span ({low}, {high}) as {alg1_pred} and Algorithm 2 recognised it as {alg2_pred}. Right now we classify for entiy {entity_name}"
                 + "\n" + f"Human analyst should look at this problem manually, for now this entity is set as {alg1_pred} as algorithm 1 is the lead"
                 + "\n" + "----------------------------------------------------------------------------------------------------------------------------------------------------------------------"+ "\n")

                intersection = alg1_pred
            else:
                intersection = "O"

            result.append(alg_result_1[alg1_idx][0])    # token
            result.append(intersection) # label
            result.append(alg_result_1[alg1_idx][2]) # lower bound of span
            result.append(alg_result_1[alg1_idx][3]) # upper bound of span
            result = tuple(result)

        overall_result.append(result)

    # If union/intersection was unsucessful -> log it to the user to investigate.
    return overall_result

def main():
    """Main MASK Framework function
               """
    print("Welcome to MASK")
    cf = Configuration()
    data = [f for f in listdir(cf.dataset_location) if isfile(join(cf.dataset_location, f))]
    algorithms = []
    # Load algorithms in data structure
    # TODO: Still optimize!
    for entity in cf.entities_list:
        # If I want to run several algos on the same entity and configuration, we will have several instances
        # of instructions in algorithms array to pass to the worker.
        for alg_name in entity["algorithm"]:
            algorithm = "ner_plugins." + alg_name
            masking_type = entity['masking_type']
            entity_name = entity['entity_name']
            if masking_type == "Redact":
                masking_class = ""
            else:
                masking_class = entity['masking_class']

            # if we have 2 models for one entity, we need to know whether we should do union or intersection of results of models:
            if "resolution" in entity:
                resolution = entity["resolution"]

            # Import the right module
            right_module = importlib.import_module(algorithm)

            # find a class and instantiate
            class_ = getattr(right_module,alg_name)

            instance = class_()
            algorithms.append({"algorithm":algorithm, "masking_type":masking_type, "entity_name":entity_name, "instance":instance, "masking_class":masking_class})

    mask_running_log = open('log_mask_running.log','w',encoding='utf-8')
    mask_running_log.write("Project name: "+cf.project_name+"\n")
    mask_running_log.write("Time of run: " + str(datetime.datetime.now()) + "\n\n")
    mask_running_log.write("RUN LOG \n")
    elements = []

    times = []
    for file in data:
        mask_running_log.write("Running stats for file: "+file+'\n')
        text = open(cf.dataset_location+"/"+file, 'r').read()
        new_text = text   # text is an original text
        overal_result = []

        time_all_1 = time.time()

        for i in range(0, len(algorithms)): # for each function call, algorithms - list of disctionaries
            alg = algorithms[i]
            next_alg = {}
            next_alg_entity_name = ""

            entity_name = alg["entity_name"]
            if i != (len(algorithms) - 1): # last instruction in the algorithms dictionary does not have any future
                next_alg = algorithms[i+1] # for the check if next algorithm's entity is the same as the current one
                next_alg_entity_name = next_alg["entity_name"]

            # if this is the case, we know that we will need to compare the results of 2 algorithms outputs.
            if entity_name == next_alg_entity_name:
                continue # as we will combine the output for entity on the next run to run masking only once instead of twice!

            else:
                previous_alg = algorithms[i-1]
                previous_alg_entity_name = previous_alg["entity_name"]

                if previous_alg_entity_name == entity_name:
                    #print(f"I am in two-algorithms completion of results for {entity_name} and algorithms {alg["algorithm"]} and {previous_alg["algorithm"]}!")
                    alg_result_1 = previous_alg["instance"].perform_NER(new_text)
                    # print(f"RESULTS OF ALGO 1 NER_BERT")
                    # print(alg_result_1)
                    alg_result_1 = consolidate_NER_results(alg_result_1, new_text) # (token, label, span_min, span_max)

                    alg_result_2 = alg["instance"].perform_NER(new_text)
                    # print(f"RESULTS OF ALGO 2 NER_CRF")
                    # print(alg_result_2)
                    alg_result_2 = consolidate_NER_results(alg_result_2, new_text) # (token, label, span_min, span_max)

                    # Do function compare_results(result1, result2) that returns overall result
                    previous_alg_name = previous_alg["algorithm"]
                    current_alg_name = alg["algorithm"]
                    print(f"alg1 is {previous_alg_name} and alg2 is {current_alg_name}")
                    overal_result = compare_results(resolution, entity_name, alg_result_1, alg_result_2, mask_running_log)
                else:
                    #print(f"I am in one-algorithms completion of results for {entity_name} and algorithm {alg["algorithm"]}!")
                    start = time.time()
                    alg_result_1 = alg["instance"].perform_NER(new_text)
                    end = time.time()

                    print("TIME FOR algorithm {} to execute NER on entity_name {} is {}".format(alg["algorithm"], entity_name, (end-start)))

                    alg_result_1 = consolidate_NER_results(alg_result_1, new_text) # (token, label, span_min, span_max)

                    overal_result = alg_result_1


            #Perform masking/redacting
            masking_type = alg["masking_type"]
            print("PROCESSED ONE ALGORITHM FUNCTIONAL CALL")

            if masking_type == "Redact":
                for i in range(0, len(overal_result)):
                    if overal_result[i][1] == entity_name:
                        token_size = overal_result[i][3]-overal_result[i][2]  # token_size = span_end - span_start
                        old_token = overal_result[i][0]
                        new_token = "XXX"
                        replacement_size = len(new_token)
                        new_text = new_text[:overal_result[i][2]] + new_token + new_text[overal_result[i][3]:]
                        token_min_span = overal_result[i][2]
                        token_max_span = overal_result[i][3]
                        overal_result = recalculate_tokens(overal_result, i, token_size, replacement_size, new_text, new_token)
                        elements.append(overal_result[i][1])
                        # We want to log spans of the identified entity to ease the job of manual identification by analysts.
                        mask_running_log.write("REDACTED ENTITY: " + overal_result[i][0] + " with span (" + str(token_min_span) + ", " + str(token_max_span) + ") " + " -- " + old_token + ' ->' + new_token + '\n')

            elif masking_type == "Mask":
                masking_class = alg["masking_class"]
                print(f"MY ENTITY NAME IS {entity_name}")
                plugin_module = importlib.import_module("masking_plugins." + masking_class)
                class_masking = getattr(plugin_module, masking_class)
                masking_instance = class_masking()
                for i in range(0, len(overal_result)):
                    if overal_result[i][1] == entity_name:
                        old_token = overal_result[i][0]
                        token_size = overal_result[i][3] - overal_result[i][2]
                        new_token = masking_instance.mask(overal_result[i][0])
                        replacement_size = len(new_token)
                        new_text = new_text[:overal_result[i][2]] + new_token + new_text[overal_result[i][3]:]
                        token_min_span = overal_result[i][2]
                        token_max_span = overal_result[i][3]
                        overal_result = recalculate_tokens(overal_result, i, token_size, replacement_size, new_text, new_token)
                        elements.append(overal_result[i][1])
                        mask_running_log.write("MASKED ENTITY: " + overal_result[i][0] + " with span (" + str(token_min_span) + ", " + str(token_max_span) + ") " + " -- " + old_token + ' ->' + new_token + '\n')

        time_all_2 = time.time()

        print("TIME to execute FULL NER FOR ONE DOCUMENT is {}".format(time_all_2-time_all_1))
        times.append((time_all_2-time_all_1))

        # Create target Directory if don't exist
        if not path.exists(cf.data_output):
            mkdir(cf.data_output)
        # write into output files
        file_handler = open(cf.data_output + "/" + file, "w")
        file_handler.write(new_text)
        file_handler.close()

        # Write a log file, there is currently a TODO: to support writing a log for 2 algorithms or more on one entity. Don't have time now to solve this issue now.
        # This functionality of logging total number of masked entities is not crucial for now.
        # for alg in algorithms:
        #     cnt = elements.count(entity_name)
        #     if masking_type == "Mask":
        #         mask_running_log.write('Total masked for '+entity_name+": "+str(cnt)+'\n')
        #     if masking_type == "Redact":
        #         mask_running_log.write('Total redacted for '+entity_name+": "+str(cnt)+'\n')

        mask_running_log.write('END for file:'+ file+'\n')
        mask_running_log.write('========================================================================')


    import numpy as np
    avg_arr = np.array(times)
    avg = np.average(avg_arr)
    std = np.std(avg_arr)
    print(f"AVERAGE TIME TO EXECUTE ON ALL NER ENTITIES FOR THE DOCUMENTS IN THE INPUT IS {avg}")
    print(f"STD OF TIME TO EXECUTE ON ALL NER ENTITIES FOR THE DOCUMENTS IN THE INPUT IS {std}")

    mask_running_log.close()


if __name__=="__main__":
    main()
