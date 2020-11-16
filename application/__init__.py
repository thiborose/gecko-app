from flask import Flask
from application.models.gector import model
import spacy 
from os import system

app = Flask(__name__, instance_relative_config=True)
app.config.from_object('config')
app.config.from_pyfile('config.py')

if app.config['DEBUG'] == True:
    try:
        from sassutils.wsgi import SassMiddleware
    except(ImportError):
        system("pip install libsass==0.20.1")
        from sassutils.wsgi import SassMiddleware
    app.wsgi_app = SassMiddleware(app.wsgi_app, {
        'application': ('static/sass', 'static/css', '/static/css')
    })

nlp = spacy.load("en_core_web_sm")

model = model.load_model(
    vocab_path = "application/models/gector/data/output_vocabulary",
    model_paths = ["application/models/gector/data/model_files/xlnet_0_gector.th"],
    model_name = "xlnet"
)

from application import routes