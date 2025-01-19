from flask import Flask, render_template, request
from model import Model
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict')
def predict_get():
    return render_template("predict.html")


@app.route('/predict', methods=["POST"])
def predict_post():
    model = Model()
    state = request.form["state"]
    type = request.form["type"]
    project = request.form["project"]
    year = int(request.form["year"])
    prediction = model.getPrediction(state, project, year, type)
    return render_template("predict.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)