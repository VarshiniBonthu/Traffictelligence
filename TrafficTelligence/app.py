import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import time
import json
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template

# Initialize the Flask app
app = Flask(__name__)


# Load the pre-trained machine learning model and scaler
model = pickle.load(open('C:/Users/VARSHINI/Downloads/model.pkl', 'rb'))
scale = pickle.load(open('C:/Users/VARSHINI/Downloads/scale.pkl', 'rb'))

with open('C:/Users/VARSHINI/Downloads/columns.json') as f:
    columns=json.load(f)
# Home route to display the homepage
@app.route('/')
def home():
   return render_template('index.html')  # Rendering the home page


# Predict route to show the predictions in a web UI
@app.route('/predict', methods=["POST"])
def predict():
    try:
        # 1. Read form input
        input_feature = [float(x) for x in request.form.values()]

        # 2. Base columns you expect (from your Colab notebook)
        names = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day', 'hours', 'minutes', 'seconds']
        input_df = pd.DataFrame([input_feature], columns=names)

        # 3. Apply same one-hot encoding as in training
        input_df = pd.get_dummies(input_df)

        # 4. Align columns to training set
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # 5. Predict
        prediction = model.predict(input_df)
        result=int(prediction[0])

        return render_template("index.html", prediction_text=f"Estimated Traffic Volume is: " + str(int(prediction[0])))

    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))


# Running the Flask app
if __name__ == "__main__":
   port = int(os.environ.get('PORT', 5000))  # Get the port number from the environment variable, default to 5000
   app.run(port=port, debug=True, use_reloader=False)  # Run the app with the specified port


