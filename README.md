# Hate Speech Detection Service

Using a hybrid algorithm (Word2Vec + Logistic Regression and N-grams string matching) I have developed a prototype of a service that being queried via HTTP with a text can predict if the text contains hate speech or not.

## Setup
```bash
virtualenv .venv -p python3 # project requires python 3.5+
source .venv/bin/activate
pip install -r requirements
cd estimator/resources/google_word2vec_model
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
cd ../../
./run
# if something doesn't work, check the config_estimator.json file and make sure the paths are set up corectly
```

You might need to find a way to slim the word2vec pre-trained model, because otherwise it will eat up huge ammouns of RAM.
One way to do it, is to limit the number of it's parameters. It can be easily done via the `config_estimator.json` file.
