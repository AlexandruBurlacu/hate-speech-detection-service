"""API Functions.

This module implements two main functions called via HTTP.

The module structure is the following:
- Function ``predict`` is called with some arguments in order
  to predict if some text contains hate speach.

- Function ``train`` is called to re-train the estimator with some new data
  which will be stored in ``./estimator/resources/feedback`` folder.

"""

# Author: Alexandru Varacuta
# Email: alexburlacu96@gmail.com


import hug  

import estimator.src.model as estimator


def predict(
              content_name: hug.types.text,
              content_data: hug.types.text,
            ):
  """JSON object with the class to which the content belongs.

  Parameters
  ----------
  content_name : str (Ex: hug.types.text)
    Content's unique name, not the title of it.

  content_data : str (Ex. hug.types.text)
    Content data expressed as a text.

  Returns
  -------
  dict
    content_name : str
      Content's unique name, not the title of it.

    content_class : {"not-hatred", "hatred"}
      The predicted class to which the queried content belongs.

  """
  predicted = "hatred" if estimator.classifier(content_data) == True else "not-hatred"
  return {
           "content_name": content_name,
           "content_class": predicted
         }


def train(n_fits: int = 10):
  """Train the available Machine Learning Estimator.

  Parameters
  ----------
  n_fits : int
    Number of partial fits to be executed over the estimator.

  Returns
  -------
  dict
    summary : dict
      The summary of the training (start_time, training_time, measured_accuracy)

  """
  summary = estimator.train_linear_clf(n_fits)
  return {
           "summary": summary
         }
