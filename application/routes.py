from flask import redirect, url_for, render_template, request
from application import app
from application import actions

@app.route("/")
def home():
    return render_template("content.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form['jsdata']
    prediction = actions.predict(user_input)
    return prediction