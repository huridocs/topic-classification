#!/usr/bin/env python
# coding: utf-8
from typing import List

import numpy
import pickle
import requests
import json

from lime.lime_text import LimeTextExplainer
from lime import submodular_pick

# load the pkl file
with open('../updated_data/UHRI_affected_persons.pkl', 'rb') as f:
    data = pickle.load(f)
    # col1: 'text' contains string of paragraph
    # last col: 'one_hot_labels' for labeled #

print('# of total examples: {}'.format(len(data)))



samples_text = data['text'].tolist()

class_i = -1  # the index for the class we focus on
specific_class = 'non-citizens'  # the class name we focus on

class_names = ['nothing', specific_class]

# finding the given class' index
for i, class_name in enumerate(class_names):
    if class_name == specific_class:
        class_i = i

print(f'The index of {specific_class} is {class_i}')


def get_labels(one_hot_labels_list):
    y = numpy.zeros(len(one_hot_labels_list))
    for i in range(len(one_hot_labels_list)):
        y[i] = one_hot_labels_list[i][class_i]
    return y


samples_classes = get_labels(data['one_hot_labels'].to_list())

explainer = LimeTextExplainer(class_names=class_names)


# SP Lime !!!


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


def easy(texts):
    return numpy.array([[1, 0] for s in texts])


sp_obj = submodular_pick.SubmodularPick(explainer, samples_text, get_predicted_labels, sample_size=1, num_features=10,
                                        num_exps_desired=1)
# num_exps_desired is the number of explanation objects returned
# num_features is maximum number of features present in explanation
# sample_size is the number of instances to explain if method == 'sample'
# ^ default method == 'sample' will sample the data uniformly at random

# shows us the features for the instances selected for one label

# add a function for a directory to store figures in
# add directories for storing figures in from different runs
for l in [exp.as_pyplot_figure(label=exp.available_labels()[0]) for exp in sp_obj.sp_explanations]:
    # l.show()
    l.savefig('fig.png')
