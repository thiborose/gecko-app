from application import nlp, model
from nltk.tokenize.treebank import TreebankWordDetokenizer
from application.models.gector.model import load_model
from application.models.gector.predict import predict_for_tokens, predict_for_string


def predict(input_string):
    tokenized_string = tokenize(input_string)
    corrected_text = predict_for_string(tokenized_string, model)
    output_text = untokenize(corrected_text)
    return output_text


def tokenize(text):
    """
        input plain text string,
        outputs tokenized string with spaces
    """
    doc = nlp(text)
    return " ".join([token.text for token in doc])

def untokenize(tokens):
    """
        input tokenized string with spaces
        outputs plain text string
    """
    return TreebankWordDetokenizer().detokenize(tokens.split(" "))