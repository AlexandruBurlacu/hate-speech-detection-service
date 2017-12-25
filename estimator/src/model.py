"""Estimator module.

This modlue contains all stuff related to the classification system.
Here are implemented both Logistic Regression based estimator and Lexicon Based.

The module structure is the following:
"""
# TODO: add description about module structure
# TODO: refactor estimator components into 2 separate modules

# Author: Alexandru Burlacu
# Email: alexburlacu96@gmail.com

from sklearn.linear_model import SGDClassifier
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

import datetime
import json
import re

import os.path


from .utils import persist, get_estimator_from_file
from .utils import get_tokens, join_paths
from .utils import clean_text

# load configurarion parameters ##############################################

PATH = os.path.abspath(os.path.dirname(__file__))

with open(join_paths(PATH, "../config_estimator.json")) as config_fp:
  """Get the estimator's configuration file.
  """
  CONFIG = json.load(config_fp)


# linear_clf #################################################################

VECTORIZER = KeyedVectors \
              .load_word2vec_format(join_paths(PATH, CONFIG["word2vec_file"]),
                             binary = True, limit = CONFIG["word2vec_limit"])


def to_word_vector(text, size = 300, model = VECTORIZER):
  """Averaged word vector infered from some text.

  Parameters
  ----------
  text : str
    String of the text to be embeded into a word vector.

  size : int, optional, default = 300
    Size of the infered word vector.

  model : gensim.models.keyedvectors
    The model which when queried, return word vectors.

  Returns
  -------
  vec : numpy.array of shape (1, 300)
    The phrase vector.

  """
  vec = np.zeros(size).reshape((1, size))
  count = 0
  for word in text:
    try:
      vec += model[word].reshape((1, size))
      count += 1
    except KeyError:
      continue
  if count != 0:
    vec /= count

  return vec

def get_data(filename):
  """Text and tags from .csv file.
  """
  with open(join_paths(PATH, CONFIG["feedback_data_dir"], filename)) as data_fp:
    data_fp.readline() # delete the head of the .csv file
    values = [tuple(line.strip().split("\t")[1:]) for line in data_fp] # `[1:]` to skip the `id` of the row
    X, y = zip(*[line for line in values if len(line) == 2])

  return X, y

def preprocess_input(source):
  """Makes data consumable by the classifier.
  """
  X_raw, y_raw = get_data(source)
  X = np.array([to_word_vector(clean_text(phrase, CONFIG)).reshape((300,)) for phrase in X_raw])
  y = np.array([int(y) for y in y_raw])

  return X, y

def train_linear_clf(n_fits):
  """Called by the `/train` HTTP GET method in `logic.py` module.

  Parameters
  ----------
  n_fits : int
    Number of partial fits applied to the linear estimator.

  Returns
  -------
  dict
    measured_accuracy : float
      Accuracy of the linear classifier after application of the specified number of partial fits.

    start_time : datetime.datetime
      Time when the function starts the process of partial fiting.

    training_time : float
      Training time in seconds.
      
  """
  clf = get_estimator_from_file(join_paths(PATH, CONFIG["model_persistence_file"])) \
        or SGDClassifier(**CONFIG["estimator"]).fit(X, y)

  start_time = datetime.datetime.now()
  [clf.partial_fit(X, y) for _ in range(n_fits)]
  end_time = datetime.datetime.now().timestamp()

  persist(clf, join_paths(PATH, CONFIG["updated_model_persistence_file"]))

  return {
          "measured_accuracy": clf.score(X_test, y_test),
          "start_time": start_time,
          "training_time": end_time - start_time.timestamp()
         }


def linear_clf_prediction(text):
  """Wrapper around `get_estimator_from_file`
  that predicts the hatefulness of the `text` argument.
  """
  clf = get_estimator_from_file(join_paths(PATH, CONFIG["model_persistence_file"]))

  return clf.predict(to_word_vector(clean_text(text, CONFIG)))


X, y = preprocess_input(CONFIG["feedback_data_file"])
X_test, y_test = preprocess_input(CONFIG["test_data_file"])

# lexicon_clf ################################################################

def hate_word_occ(ordered_bow, hate_grams):
  """Number of hate grams in the given bag-of-words.
  """
  score = 0
  ordered_bow = list(ordered_bow)
  bow_counted = {gram: ordered_bow.count(gram) for gram in ordered_bow}
  for term in hate_grams:
    token = bow_counted.get(term)
    score += token if token != None else 0

  return score

def lexic_score(text):
  """Number of 1, 2 and 3 grams related to hate speach.
  """
  hate_words = lambda gs: hate_word_occ(gs, get_tokens(CONFIG["hate_words_file"]))

  gram_1 = clean_text(text, CONFIG)

  gram_2 = map(lambda x, y: " ".join([x, y]),
                 gram_1,
                 list(gram_1)[1:])

  gram_3 = map(lambda x, y, z: " ".join([x, y, z]),
                 gram_1,
                 list(gram_1)[1:],
                 list(gram_1)[2:])

  return map(hate_words, [gram_1, gram_2, gram_3])

def classifier(text, k1 = 0.6, k2 = 0.4):
  """Combines predictions of lexic classifier and linear one.
  """
  return (sum(lexic_score(text)) * k1 + (not linear_clf_prediction(text)) * k2) > 0.6
