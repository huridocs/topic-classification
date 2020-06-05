#!/usr/bin/env python
# coding: utf-8

import numpy
import pickle
import requests
from requests.adapters import HTTPAdapter
import json
import matplotlib
import matplotlib.pyplot as plt

from lime.lime_text import LimeTextExplainer
from lime import submodular_pick

# load the pkl file
with open('../updated_data/UHRI_affected_persons.pkl', 'rb') as f:
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

explainer = LimeTextExplainer(class_names=class_names)

# SP Lime !!!

def requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None):
  session = session or requests.Session()
  retry = requests.packages.urllib3.util.retry.Retry(
    total=retries,
    read=retries,
    connect=retries,
    backoff_factor=backoff_factor,
    status_forcelist=status_forcelist,
  )
  adapter = HTTPAdapter(max_retries=retry)
  session.mount('http://', adapter)
  session.mount('https://', adapter)
  return session

def get_predicted_labels(texts: str):
  request_adapter = requests_retry_session()

  data = dict()
  data['samples'] = [{"seq": s} for s in texts]

  response = request_adapter.post(url='http://localhost:5005/classify',
                                  headers={'Content-Type': 'application/json'},
                                  params=(('model', 'affected_persons'),),
                                  data=json.dumps(data))

  if response.status_code != 200: # checking if the http request failed
    print('failed try again?')
    return numpy.zeros((len(data), 2))
  result = json.loads(response.text)

  full_results = numpy.zeros((len(data), 2))
  for index, sample in enumerate(result['samples']):
    if 'non_citizens' in str(sample['predicted_labels']):
      full_results[index][1] = 1
    else:
      full_results[index][0] = 1

  return full_results

def easy(texts):
  return numpy.array([[1, 0] for s in texts])


sp_obj = submodular_pick.SubmodularPick(explainer, samples_text, get_predicted_labels, sample_size=1, num_features=10, num_exps_desired=1)
# num_exps_desired is the number of explanation objects returned
# num_features is maximum number of features present in explanation
# sample_size is the number of instances to explain if method == 'sample'
# ^ default method == 'sample' will sample the data uniformly at random

# shows us the features for the instances selected for one label

# add a function for a directory to store figures in
# add directories for storing figures in from different runs
for l in [exp.as_pyplot_figure(label=exp.available_labels()[0]) for exp in sp_obj.sp_explanations]:
  #l.show()
  l.savefig('fig.png')

