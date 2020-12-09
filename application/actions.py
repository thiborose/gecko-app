from application import nlp, model
from nltk.tokenize.treebank import TreebankWordDetokenizer
from application.models.gector.model import load_model
from application.models.gector.predict import predict_for_sentences
from application.models.gector.utils.preprocess_data import align_sequences

def predict(input_text: str) -> str:
    """Returns the predicted correction for an input text."""

    tokenized_sentences = tokenize_and_segment(input_text)
    corrected_sentences = predict_for_sentences(tokenized_sentences, model)
    output_text = untokenize(corrected_sentences)
    # sent_with_tags = align_sequences(input_string, output_text)
    return output_text 


def tokenize_and_segment(input_text: str) -> list:
    """Returns a list of tokenized sentences (strings)."""

    doc = nlp(input_text)
    sentences = []
    for sent in doc.sents:
        sentences.append(' '.join(token.text for token in sent))
    return sentences

def untokenize(sentences: list) -> str:
    return ' '.join(TreebankWordDetokenizer().detokenize(sent) for sent in sentences)