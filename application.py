import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize the Flask app
application = Flask(__name__)
app = application

# Load the trained Lasso model and the scaler
new_model = pickle.load(open('models/new_model.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Creating the prediction route
@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        # Collect all form data ( use only the 8 model features)
        temperature = float(request.form['Temperature'])  # Air temperature in Celsius
        rh = float(request.form['RH'])                   # Relative Humidity in %
        ws = float(request.form['Ws'])                   # Wind speed in km/h
        rain = float(request.form['Rain'])               # Rainfall in mm/m²
        ffmc = float(request.form['FFMC'])               # Fine Fuel Moisture Code
        dmc = float(request.form['DMC'])                 # Duff Moisture Code
        dc = float(request.form['DC'])                   # Drought Code
        isi = float(request.form['ISI'])                 # Initial Spread Index
            

        # Use only model-trained features
        input_features = [temperature, rh, ws, rain, ffmc, dmc, dc, isi]

        # Transform and predict
        scaled_features = standard_scaler.transform([input_features])
        prediction = new_model.predict(scaled_features)[0]

        return render_template('home.html', prediction=round(prediction, 2))
    
    return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
    