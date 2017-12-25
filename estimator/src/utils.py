"""Estimator utility functions.

This module contains some utility functions used by the ``model`` module
where the estimator is implemented.

The module structure is the following:
- Function ``get_tokens`` used to parse comma separated strings from some file
  and return a list of strings. It is used in ``model`` file to extract
  stop words and hate words from files.

- Function ``persist`` is a wrapper around sklearn's ``joblib.dump`` function,
  optimized for pickling big numpy arrays.

- Function ``get_estimator_from_file`` is a wrapper around
  sklearn's ``joblib.load`` function, which in case of a ``FileNotFoundError``
  will return ``None``.

"""

# Author: Alexandru Varacuta <alexburlacu96@gmail.com>
# LikedIn: linkedin.com/in/alexandru-varacuta-0b68ba139


from sklearn.externals import joblib
import os.path
import re

PATH = os.path.abspath(os.path.dirname(__file__))

def join_paths(*args):
  return os.path.join(*args)

def get_tokens(filename):
  """List of tokens from the given file.

  Parameters
  ----------
  filename : string
    The name of the file in which some tokens separated by commas are stored.

  Returns
  -------
  words : list of str
    List with the tokens from the given file.

  """
  words = None
  with open(PATH + "/" + filename) as fp:
    _words = [line.split(",") for line in fp.readlines()]
    words = _words[0][:-1] # because of the newline character

    return words

def persist(estimator, filename):
  """Save trained model on disk.

  Parameters
  ----------
  estimator : sklearn.base.BaseEstimator
    Estimator to be stored on disk.

  filename : str
    The name of the file in which the estimator will be stored.

  Returns
  -------
  None
  """
  joblib.dump(estimator, filename)

def get_estimator_from_file(filename):
  """Retrives the estimator from a file if exists.

  Parameters
  ----------
  filename : str
    The name of the file in which the estimator is stored.

  Returns
  -------
  sklearn.base.BaseEstimator or None
    If the file does not exist or any exception happens during it's load,
    None will be returned.
  """
  try:
    return joblib.load(filename)
  except:
    return None


def _rm_punctuation(string):
  return re.sub(r'[^\w\s]', '', string)

def clean_text(corpus, config):
  """Get tokens of a phrase string and
  clean it from punctuation and stop words.
  """
  not_stop_word = lambda t: not t in get_tokens(config["stop_words_file"])

  no_punct = (_rm_punctuation(s).lower() for s in corpus.split())

  return (token for token in no_punct if not_stop_word(token))