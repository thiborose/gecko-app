from flask import Flask
from application.models.gector import model
import spacy 

from sassutils.wsgi import SassMiddleware

app = Flask(__name__, instance_relative_config=True)
app.config.from_object('config')
app.config.from_pyfile('config.py')

if app.config['DEBUG'] == True:
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