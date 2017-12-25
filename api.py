"""API Endpoints

This module contains the entrypoint for the API of
the hate speech detection system.

- The ``config`` dictionary contains some key:value pairs
  to change the behavior of the Hug api.

GET /prediction - given a sentence in the query, returns the prediction hate/not hate speech
GET /training - given the number of epochs, trains the NLP model
"""

# Author: Alexandru Burlacu
# Email: alexburlacu96@gmail.com

import logic
import hug

api = hug.API(__name__)

config = {
  "api": api,
  "on_invalid": hug.redirect.not_found,
  "output_format": hug.output_format.pretty_json
}

hug.get("/predict", **config)(logic.predict)

hug.get("/training", **config)(logic.train)
