from flask import Flask
from application.models.gector import model
import application.models.sentence_reorder as sentence_reoder
from os import listdir

import re

app = Flask(__name__, instance_relative_config=True)

# setting up environment
root_dir = listdir()
# print(root_dir)
if "config.py" in root_dir:
    app.config.from_object('config')
else:
    app.config['DEBUG'] = False


# if app.config['DEBUG'] == True:
#     try:
#         from sassutils.wsgi import SassMiddleware
#     except(ImportError):
#         system("pip install libsass==0.20.1")
#         from sassutils.wsgi import SassMiddleware
#     app.wsgi_app = SassMiddleware(app.wsgi_app, {
#         'application': ('static/sass', 'static/css', '/static/css')
#     })

model = model.load_model(
    vocab_path = "application/models/gector/data/output_vocabulary",
    model_paths = ["application/models/gector/data/model_files/xlnet_0_gector.th"],
    model_name = "xlnet"
)

sentence_reoder.load_model()

DELIMITER = 'SEPL|||SEPR'
RE_HYPHENS = re.compile(r'(\w) - (\w)')
RE_QUOTES1 = re.compile(r"([\"']) (.*?[^\\])")
RE_QUOTES2 = re.compile(r"(.*?[^\\]) ([\"'])")

from application import routes
