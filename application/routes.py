from flask import redirect, url_for, render_template, request
from application import app
from application import actions

# serve the home page
@app.route("/")
def home():
    return render_template("content.html")

# response to the ajax predict query
@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form['input_text']
    reorder = request.form['reorder']
    reorder = True if reorder=="true" else False
    prediction = actions.predict(user_input, reorder)
    return prediction