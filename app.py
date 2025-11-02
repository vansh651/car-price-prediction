from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
try:
    model = joblib.load('model.sav')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Create a simple backup model in case the saved model isn't available
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        if request.form:
            present_price = float(request.form.get('present_price', 0))
            kms_driven = int(request.form.get('kms_driven', 0))
            fuel_type = request.form.get('fuel_type', '').upper()
            seller_type = request.form.get('seller_type', '').upper()
            transmission = request.form.get('transmission', '').upper()
            owner = int(request.form.get('owner', 0))
            year = int(request.form.get('year', 2020))
        # Get data from JSON (API requests)
        else:
            data = request.get_json()
            present_price = float(data.get('present_price', 0))
            kms_driven = int(data.get('kms_driven', 0))
            fuel_type = data.get('fuel_type', '').upper()
            seller_type = data.get('seller_type', '').upper()
            transmission = data.get('transmission', '').upper()
            owner = int(data.get('owner', 0))
            year = int(data.get('year', 2020))
        
        # Convert categorical inputs to numeric
        fuel_dict = {'PETROL': 0, 'DIESEL': 1, 'CNG': 2}
        seller_dict = {'DEALER': 0, 'INDIVIDUAL': 1}
        transmission_dict = {'MANUAL': 0, 'AUTOMATIC': 1}
        
        fuel_type_num = fuel_dict.get(fuel_type, 0)
        seller_type_num = seller_dict.get(seller_type, 0)
        transmission_num = transmission_dict.get(transmission, 0)
        age = 2025 - year
        
        # Create features array for prediction
        features = np.array([[
            present_price, 
            kms_driven, 
            fuel_type_num, 
            seller_type_num, 
            transmission_num, 
            owner, 
            age
        ]])
        
        # Make prediction
        predicted_price = model.predict(features)[0]
        
        # Return prediction based on request type
        if request.form:
            return render_template('index.html', 
                                  prediction=round(float(predicted_price), 2),
                                  form_data={
                                      'present_price': present_price,
                                      'kms_driven': kms_driven,
                                      'fuel_type': fuel_type,
                                      'seller_type': seller_type,
                                      'transmission': transmission,
                                      'owner': owner,
                                      'year': year
                                  })
        else:
            return jsonify({
                'success': True,
                'predicted_price': round(float(predicted_price), 2),
                'message': f"Predicted Resale Price: ₹{predicted_price:.2f} lakhs"
            })
        
    except Exception as e:
        if request.form:
            return render_template('index.html', error=str(e))
        else:
            return jsonify({
                'success': False,
                'error': str(e),
                'message': 'Failed to make prediction. Please check your input data.'
            })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Return information about the model"""
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_.tolist()
        features = ['Present Price', 'Kms Driven', 'Fuel Type', 'Seller Type', 'Transmission', 'Owner', 'Age']
        
        importance_pairs = list(zip(features, feature_importance))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return render_template('model_info.html', 
                              feature_importance=importance_pairs,
                              model_type='Random Forest Regressor')
    else:
        return render_template('model_info.html',
                              model_type='Machine Learning Model',
                              message='Feature importance not available for this model')

if __name__ == '__main__':
    app.run(debug=True)
