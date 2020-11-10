from flask import Flask
from application.models.gector import model
import spacy 

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")

model = model.load_model(
    vocab_path = "application/models/gector/data/output_vocabulary",
    model_paths = ["application/models/gector/data/model_files/xlnet_0_gector.th"],
    model_name = "xlnet"
)

from application import routes