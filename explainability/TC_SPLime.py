#!/usr/bin/env python
# coding: utf-8

import numpy
import csv
import os
from datetime import datetime
import pickle
import requests
import json
import matplotlib.pyplot as plt
import lime

from lime.lime_text import LimeTextExplainer
from lime import submodular_pick
from random import choice
from typing import List


def set_labels(specific_class: str, labels_list: List[str]) -> List[int]:
    classes_list = []
    for index, sample_labels in enumerate(labels_list):
        if specific_class in sample_labels:
            classes_list.append(1)
        else:
            classes_list.append(0)
    return classes_list


def count_labels_in_specific_class(classes_list: List[int]) -> int:
    count = 0
    for index, sample_class in enumerate(classes_list):
        if sample_class == 1:
            count += 1
    return count


def get_samples_indices_within_wc(max_word_count_per_text: int, samples_lengths: List[int]) -> List[int]:
    indices_ok_samples = []
    for index, length in enumerate(samples_lengths):
        if length <= max_word_count_per_text:
            indices_ok_samples.append(index)
    return indices_ok_samples


def get_predicted_labels(texts: List[str]):
    batch_size = 10
    print(f'Predicting {len(texts)} samples')
    classes_predictions = list()

    samples_batch = []
    for index, text in enumerate(texts):
        samples_batch.append(text)

        if len(samples_batch) == batch_size:
            sample_to_send = {'samples': [{"seq": t} for t in samples_batch]}
            response = requests.post(url='http://localhost:5005/classify',
                                     headers={'Content-Type': 'application/json'},
                                     params=(('model', 'affected_persons'),),
                                     data=json.dumps(sample_to_send))

            result = json.loads(response.text)['samples']
            results = [[1, 0] if 'non_citizens' in str(each_result['predicted_labels'])
                       else [0, 1] for each_result in result]

            classes_predictions.extend(results)
            samples_batch = []

    return numpy.array(classes_predictions)


# Function mainly for debugging the prediction function
def easy(texts: List[str]) -> numpy.array:
    full_predicted_labels = []
    for text in texts:
        first_label = choice([0, 1])
        second_label = 1 if first_label == 0 else 0
        predicted_labels = [first_label, second_label]
        full_predicted_labels.append(predicted_labels)
    return numpy.array(full_predicted_labels)


def get_run_path() -> str:
    time = datetime.now().strftime("%Y%m%d-%H%M%S%f")
    run_num = f'Run_{time}'
    return run_num


def save_explanations_as_figs(sp_obj, run_path):
    # define the name of the directory to be created
    path = f'figures_saved/{run_path}'
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
        #  stores the figures in the path given
        index = 0
        for fig in [exp.as_pyplot_figure(label=exp.available_labels()[0]) for exp in sp_obj.sp_explanations]:
            fig_path = f'{path}/fig{index}.png'
            fig.savefig(fig_path)
            index += 1


def get_splime_explanations(sp_explanations) -> List[any]:
    print(f'The number of sp_explanations is: {len(sp_explanations)}')
    class1_explanations = []
    for exp in sp_explanations:
        if exp.available_labels()[0] == 1:
            class1_explanations.append(exp.as_list())
    return class1_explanations


def save_explanations_to_csv(focused_class: str, csv_path: str, exps, specified_num_features: int):
    # saves the explanation #, features, and values to a csv file
    exp_list_for_csv = []
    for index, exp in enumerate(exps):
        # TODO: get all features on the same row
        for feature_index in range(specified_num_features):
            exp_list_for_csv.append([focused_class, index, exp[feature_index][0], exp[feature_index][1]])
    outfile = open(csv_path, 'w')
    writer = csv.writer(outfile)
    writer.writerow(["Class", "Explanation #", "Features", "Values"])
    writer.writerows(exp_list_for_csv)
    outfile.close()


def run_splime_for_TC(word_count_max: int, focus_class: str, sp_samples_num: int,
                      sp_features_num: int, sp_explanations_num: int, save_figs: bool):
    # load the pkl file
    with open('../updated_data/UHRI_affected_persons.pkl', 'rb') as f:
        data = pickle.load(f)
        # col1: 'text' contains string of paragraph
        # col2: 'affected_persons' contains string of labels for a sample
        # col3: 'seq_length' tells us how many words is in a given sample
        # col4: 'label' is a list of str labels for a sample
        # col5: 'str_label' is one whole string with all labels for a sample
        # col6: 'one_hot_labels' for labeled #

    specific_class = focus_class  # the class name we focus on
    samples_texts = data['text'].tolist()
    samples_labels = data['label'].to_list()
    samples_classes = set_labels(specific_class, samples_labels)
    samples_lengths = data['seq_length'].to_list()

    print(f'\nThe total # of samples: {len(data)}')
    print(f'The class we focus on is: {specific_class}')
    print(f'The total number of times that class is in the data is: {count_labels_in_specific_class(samples_classes)}')
    print(f'The longest word count in a sample is: {max(samples_lengths)}')

    indices_list_ok_samples = get_samples_indices_within_wc(word_count_max, samples_lengths)

    print(f'\nThe specified max word count (wc) is: {word_count_max}')
    print(f'The number of samples within that wc is: {len(indices_list_ok_samples)}')

    updated_samples_texts = [samples_texts[i] for i in indices_list_ok_samples]
    updated_samples_classes = [samples_classes[i] for i in indices_list_ok_samples]

    print(f'\nThe updated data has this many samples in it: {len(updated_samples_texts)}')
    print(f'The # of times the specified class is in the updated data is: {count_labels_in_specific_class(updated_samples_classes)}')

    class_names = ['other', specific_class]
    explainer = LimeTextExplainer(class_names=class_names)

    print('\nStarting SP-Lime!!!')
    sp_obj = submodular_pick.SubmodularPick(explainer, updated_samples_texts, get_predicted_labels,
                                            sample_size=sp_samples_num, num_features=sp_features_num,
                                            num_exps_desired=sp_explanations_num)
    # num_exps_desired is the number of explanation objects returned per class
    # num_features is maximum number of features present in explanation
    # sample_size is the number of instances to explain

    run_path = get_run_path()
    if save_figs:
        save_explanations_as_figs(sp_obj, run_path)
    sp_explanations_list = get_splime_explanations(sp_obj.sp_explanations)

    return sp_explanations_list, run_path


if __name__ == "__main__":
    # NOTE if running for first time on machine
    # add two empty directories to the explainability folder: "csv_saved" and "figures_saved"

    # specify parameters
    max_wc = 40
    class_wanted = 'non-citizens'
    num_samples = 2
    num_features = 4
    num_exps = 2
    save_figures = True

    explanations, path_to_run = run_splime_for_TC(max_wc, class_wanted, num_samples, num_features, num_exps, save_figures)

    print('\nBack in main function now...')
    print(explanations[0])  # print the features and their values for the first explanation

    # save explanation information into csv
    path_for_csv = f'csv_saved/{path_to_run}.csv'
    save_explanations_to_csv(class_wanted, path_for_csv, explanations, num_features)