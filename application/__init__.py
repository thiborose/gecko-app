from flask import Flask
from application.models.gector import model
import application.models.sentence_reorder as sentence_reoder
import spacy 
from os import system, listdir

import re

app = Flask(__name__, instance_relative_config=True)

# setting up environment
root_dir = listdir()

# If the config file is not present, then it's because Docker doesn't see it (.dockerignore)
# Then the configuration is set to production mode (deployment of the app)
if "config.py" in root_dir:
    app.config.from_object('config')
else:
    app.config['ENV'] = "prod"
    app.config['DEBUG'] = False

# If debug mode (dev environment), load sass-middleware
# To compile SCSS stylesheets in real time
if app.config['DEBUG'] == True:
    try:
        from sassutils.wsgi import SassMiddleware
    except(ImportError):
        system("pip install libsass==0.20.1")
        from sassutils.wsgi import SassMiddleware
    app.wsgi_app = SassMiddleware(app.wsgi_app, {
        'application': ('static/sass', 'static/css', '/static/css')
    })

# Loading spacy's pipeline
nlp = spacy.load("en_core_web_sm")

# Loading Gector
model = model.load_model(
    vocab_path = "application/models/gector/data/output_vocabulary",
    model_paths = ["application/models/gector/data/model_files/xlnet_0_gector.th"],
    model_name = "xlnet"
)

# Loading the sentence reordering model
sentence_reoder.load_model()


# Patterns to export
DELIMITER = 'SEPL|||SEPR'
RE_HYPHENS = re.compile(r'(\w) - (\w)')
RE_QUOTES1 = re.compile(r"([\"']) (.*?[^\\])")
RE_QUOTES2 = re.compile(r"(.*?[^\\]) ([\"'])")
RE_QUOTES = re.compile(r"([\"']) (.+) ([\"'])")

from application import routes
