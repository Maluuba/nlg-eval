
'''
@article{Mir2019EvaluatingST,
  title={Evaluating Style Transfer for Text},
  author={Remi Mir and Bjarke Felbo and Nick Obradovich and Iyad Rahwan},
  journal={ArXiv},
  year={2019},
  volume={abs/1904.02295}
}

Implementation and Pretrained Classifier models from the repo: https://github.com/passeul/style-transfer-model-evaluation

This code can be used to evaluate the naturalness of output sentiment texts of examined style transfer models.

For a baseline understanding of what is considered "natural," any method used for automated evaluation of naturalness 
also requires an understanding of the human-sourced input texts. 

Inspired by the adversarial evaluation approach in "Generating Sentences from a Continuous Space"
(Bowman et al., 2016), we trained unigram logistic regression classifiers and LSTM logistic regression classifiers 
on samples of input texts and output texts for each style transfer model.

Via adversarial evaluation, the classifiers must distinguish human-generated inputs from machine-generated outputs. 
The more natural an output is, the likelier it is to fool an adversarial classifier.

We calculate percent agreement with human judgments. Both classifiers show greater agreement on which texts are
considered more natural with humans given relative scoring tasks than with those given absolute scoring tasks.


'''
import joblib
import re
from keras.models import load_model as load_keras_model
from keras.preprocessing.sequence import pad_sequences
from nlgeval.naturalness.tokenizer import *

RE_PATTERN = re.compile(r'|'.join(IGNORED) + r'|(' + r'|'.join(TOKENS) + r')',
                        re.UNICODE)
CLASSIFIER_BASE_PATH = '../models/naturalness_classifiers'
# adjust vocabulary to account for unknowns
def load_model(path):
    return joblib.load(path)
def invert_dict(dictionary):
    return dict(zip(dictionary.values(), dictionary.keys()))

MAX_SEQ_LEN = 30 # for neural classif
TEXT_VECTORIZER = load_model('../models/vectorizer.pkl')
VOCABULARY = TEXT_VECTORIZER.vocabulary_
INVERSE_VOCABULARY = invert_dict(VOCABULARY)
VOCABULARY[INVERSE_VOCABULARY[0]] = len(VOCABULARY)
VOCABULARY['CUSTOM_UNKNOWN'] = len(VOCABULARY)+1

class NaturalnessClassifier():
    def __init__(self, style_transfer_model_name):
        self.path = f'{CLASSIFIER_BASE_PATH}/neural_{style_transfer_model_name}.h5'
        self.classifier = load_keras_model(self.path)

    def score(self, texts):
        print(texts)
        inps = format_inputs(texts)
        distribution = self.classifier.predict(inps)
        scores = distribution.squeeze()
        return scores

def format_inputs(texts):
    
    # prepare texts for use in neural classifier
    texts_as_indices = []
    for text in texts:
        convert_to_indices(text)
        texts_as_indices.append(convert_to_indices(text))
    return pad_sequences(texts_as_indices, maxlen=MAX_SEQ_LEN, padding='post', truncating='post', value=0.)

def convert_to_indices(text):
    # tokenize input text
    tokens = re.compile(RE_PATTERN).split(text)    
    non_empty_tokens = list(filter(lambda token: token, tokens))    
    indices = []
    # collect indices of tokens in vocabulary
    for token in non_empty_tokens:
        if token in VOCABULARY:
            index = VOCABULARY[token]
        else:
            index = VOCABULARY['CUSTOM_UNKNOWN']
        indices.append(index)
    return indices