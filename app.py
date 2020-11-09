from flask import Flask, redirect, url_for, render_template, request
import difflib
from gector.model import load_model
from gector.predict import predict_for_tokens
import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer

from sassutils.wsgi import SassMiddleware


app = Flask(__name__)

app.wsgi_app = SassMiddleware(app.wsgi_app, {
    'app': ('static/sass', 'static/css', '/static/css')
})

nlp = spacy.load("en_core_web_sm")


@app.route("/")
def home():
    return render_template("content.html")


class Gec():
    def __init__(
        self, vocabulary_path = './gector/data/output_vocabulary/',
        model_path = ['./static/nn_models/xlnet_0_gector.th'],
        model_name = "xlnet"):


        self.model = load_model(vocab_path= vocabulary_path,
                   model_paths=model_path,
                   model_name=model_name)
        
    def predict(self, input_string):
        tokens = self.tokenize(input_string)
        corrected_tokens = predict_for_tokens(tokens, self.model)
        output_text = self.untokenize(corrected_tokens)
        return output_text
    
    def tokenize(self, text):
        """
        input plain text string,
        outputs tokenized string with spaces
        """
        doc = nlp(text)
        return [token.text for token in doc]

    def untokenize(self, tokens):
        """
        input tokenized string with spaces
        outputs plain text string
        """
        return TreebankWordDetokenizer().detokenize(tokens)

    def show_diff(self, text, n_text):
        """
        compares two strings
        gives the correct css classes accordingly
        """
        seqm = difflib.SequenceMatcher(None, text, n_text)
        output= []
        for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
            if opcode == 'equal':
                output.append(seqm.a[a0:a1])
            elif opcode == 'insert':
                output.append('<span class="delta-insert">' + seqm.b[b0:b1] + '</span>')
            elif opcode == 'delete':
                output.append('<span class="delta-delete">' + seqm.a[a0:a1] + '</span>')
            elif opcode == 'replace':
                output.append('<span class="delta-replace">' + seqm.b[b0:b1] + "</span>")
            else:
                raise RuntimeError("unexpected opcode")
        return ''.join(output)



@app.route("/predict")
def predict():
    user_input = request.args.get('jsdata')
    output = gec.predict(user_input)
    return gec.show_diff(user_input, output)



if __name__ == "__main__":
    gec = Gec()
    app.run(debug = True)
