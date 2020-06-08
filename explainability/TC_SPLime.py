#!/usr/bin/env python
# coding: utf-8
from typing import List

import numpy
import pandas as pd
import pickle
import requests
import json

from lime.lime_text import LimeTextExplainer
from lime import submodular_pick

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
            results = [[1, 0] if 'non_citizens' in str(each_result['predicted_labels']) else [0, 1] for each_result in result]

            classes_predictions.extend(results)
            samples_batch = []

    return numpy.array(classes_predictions)

# load the pkl file
with open('../updated_data/UHRI_affected_persons.pkl', 'rb') as f:
    data = pickle.load(f)
    # col1: 'text' contains string of paragraph
    # col2: 'affected_persons' contains string of labels for a sample
    # col3: 'seq_length' tells us how many words is in a given sample
    # col4: 'label' is a list of str labels for a sample
    # col5: 'str_label' is one whole string with all labels for a sample
    # col6: 'one_hot_labels' for labeled #

specific_class = 'non-citizens'  # the class name we focus on
samples_texts = data['text'].tolist()
samples_labels = data['label'].to_list()
samples_classes = set_labels(specific_class, samples_labels)
samples_lengths = data['seq_length'].to_list()

print(f'\nThe total # of samples: {len(data)}')
print(f'The class we focus on is: {specific_class}')
print(f'The total number of times that class is in the data is: {count_labels_in_specific_class(samples_classes)}')
print(f'The longest word count in a sample is: {max(samples_lengths)}')

max_word_count = 50
indices_list_ok_samples = []
for index, length in enumerate(samples_lengths):
    if length <= max_word_count:
        indices_list_ok_samples.append(index)

print(f'\nThe specified max word count (wc) is: {max_word_count}')
print(f'The number of samples within that wc is: {len(indices_list_ok_samples)}')

updated_samples_texts = [samples_texts[i] for i in indices_list_ok_samples]
updated_samples_classes = [samples_classes[i] for i in indices_list_ok_samples]

print(f'\nThe updated data has this many samples in it: {len(updated_samples_texts)}')
print(f'The # of times the specified class is in the updated data is: {count_labels_in_specific_class(updated_samples_classes)}')

class_names = ['nothing', specific_class]
explainer = LimeTextExplainer(class_names=class_names)

print('\nStarting SP-Lime!!!')
print('Go to the terminal where the server is running to get further updates...')
sp_obj = submodular_pick.SubmodularPick(explainer, updated_samples_texts, get_predicted_labels, sample_size=2, num_features=5, num_exps_desired=1)
# num_exps_desired is the number of explanation objects returned
# num_features is maximum number of features present in explanation
# sample_size is the number of instances to explain if method == 'sample'
# ^ default method == 'sample' will sample the data uniformly at random

#TODO: check this...not yet sure if it works
df = pd.DataFrame({})
for this_label in range(2):
    dfl = []
    for i, exp in enumerate(sp_obj.sp_explanations):
        l = exp.as_list(label=this_label)
        l.append(("exp number", i))
        dfl.append(dict(l))
    dftest = pd.DataFrame(dfl)
    df = df.append(pd.DataFrame(dfl, index=[class_names[this_label] for i in range(len(sp_obj.sp_explanations))]))
print(df)
