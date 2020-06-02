#!/usr/bin/env python
# coding: utf-8

import lime
import sklearn
import sklearn.ensemble
import sklearn.metrics
import sys
import numpy
import pandas
import nltk
import pickle

# import python modules
from utils import modeling, optimization, tokenization
from utils.analysis import plot_category_distribution

# Connect to google cloud
from google.colab import auth

from explainability import get_predicted_label

auth.authenticate_user()

#@title Load
from load import load_data, load_unique_labels
from utils import io

# load the pkl file
with open('updated_data/UHRI_affected_persons.pkl', 'rb') as f:
  data = pickle.load(f)
  #col1: 'text' contains string of paragraph
  #last col: 'one_hot_labels' for labeled #

print('# of total examples: {}'.format(len(data)))

samples_text = data['text'].tolist()

class_i = -1 # the index for the class we focus on
specific_class = 'non-citizens' # the class name we focus on

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

from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)

# SP Lime !!! 
from lime import submodular_pick

# add get_predicted_label method to be used for sp-lime, so we can use our own model


sp_obj = submodular_pick.SubmodularPick(explainer, samples_classes, get_predicted_label, sample_size=100, num_features=10, num_exps_desired=20) # method='full'
# can add "method='full'" to get explanations from entire data
# num_exps_desired is the number of explanation objects returned
# num_features is maximum number of features present in explanation
# sample_size is the number of instances to explain if method == 'sample'
# ^ default method == 'sample' will sample the data uniformly at random

# shows us the features for the instances selected for one label
[exp.as_pyplot_figure(label=exp.available_labels()[0]) for exp in sp_obj.sp_explanations]; 

