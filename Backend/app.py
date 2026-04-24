from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# ============================================
# MODEL PATHS
# ============================================
MODEL_PATH = 'models'

# Citizen Segmentation
CITIZEN_MODEL_PATH = os.path.join(MODEL_PATH, 'citizen_segment_model.pkl')
CITIZEN_ENCODER_PATH = os.path.join(MODEL_PATH, 'label_encoders.pkl')

# House Price
HOUSE_MODEL_PATH = os.path.join(MODEL_PATH, 'house_price_model.pkl')
HOUSE_FEATURES_PATH = os.path.join(MODEL_PATH, 'house_price_features.pkl')

# Traffic
TRAFFIC_MODEL_PATH = os.path.join(MODEL_PATH, 'traffic_model.pkl')
TRAFFIC_SCALER_PATH = os.path.join(MODEL_PATH, 'traffic_scaler.pkl')
TRAFFIC_FEATURES_PATH = os.path.join(MODEL_PATH, 'traffic_features.pkl')
TRAFFIC_ENCODER_PATH = os.path.join(MODEL_PATH, 'traffic_label_encoders.pkl')

# Crime
CRIME_MODEL_PATH = os.path.join(MODEL_PATH, 'crime_model.pkl')
CRIME_SCALER_PATH = os.path.join(MODEL_PATH, 'crime_scaler.pkl')
CRIME_ENCODER_PATH = os.path.join(MODEL_PATH, 'crime_label_encoders.pkl')

# ============================================
# LOAD ALL MODELS
# ============================================
print("=" * 60)
print("🚀 SMART CITY AI BACKEND INITIALIZING...")
print("=" * 60)

# Load Citizen Model
try:
    citizen_model = joblib.load(CITIZEN_MODEL_PATH)
    citizen_encoders = joblib.load(CITIZEN_ENCODER_PATH)
    print("✅ Citizen Segmentation Model Loaded")
except Exception as e:
    print(f"❌ Citizen Model Error: {e}")
    citizen_model = None
    citizen_encoders = None

# Load House Price Model
try:
    house_model = joblib.load(HOUSE_MODEL_PATH)
    try:
        house_features = joblib.load(HOUSE_FEATURES_PATH)
    except:
        house_features = ['Square_Footage', 'Num_Bathrooms', 'Num_Bedrooms', 'Lot_Size', 'Neighborhood_Quality']
    print("✅ House Price Model Loaded")
except Exception as e:
    print(f"❌ House Price Error: {e}")
    house_model = None
    house_features = None

# Load Traffic Model
try:
    traffic_model = joblib.load(TRAFFIC_MODEL_PATH)
    traffic_scaler = joblib.load(TRAFFIC_SCALER_PATH)
    traffic_features = joblib.load(TRAFFIC_FEATURES_PATH)
    try:
        traffic_encoders = joblib.load(TRAFFIC_ENCODER_PATH)
    except:
        traffic_encoders = None
    print("✅ Traffic Model Loaded")
except Exception as e:
    print(f"❌ Traffic Model Error: {e}")
    traffic_model = None
    traffic_scaler = None
    traffic_features = None
    traffic_encoders = None

# Load Crime Model
try:
    crime_model = joblib.load(CRIME_MODEL_PATH)
    crime_scaler = joblib.load(CRIME_SCALER_PATH)
    try:
        crime_encoders = joblib.load(CRIME_ENCODER_PATH)
    except:
        crime_encoders = None
    print("✅ Crime Model Loaded")
except Exception as e:
    print(f"❌ Crime Model Error: {e}")
    crime_model = None
    crime_scaler = None
    crime_encoders = None

print("=" * 60)
print("🎯 ALL MODELS LOADED SUCCESSFULLY!")
print("=" * 60)

# ============================================
# HELPER FUNCTIONS
# ============================================

def encode_citizen_data(data):
    """Encode categorical data for citizen model"""
    df = pd.DataFrame([data])
    for col in citizen_encoders:
        if col in df.columns:
            try:
                df[col] = citizen_encoders[col].transform(df[col])
            except:
                # Handle unknown categories
                df[col] = citizen_encoders[col].transform([df[col].iloc[0]])[0]
    return df

def encode_traffic_data(data):
    """Encode categorical data for traffic model using hardcoded encoders"""
    from sklearn.preprocessing import LabelEncoder

    # Hardcoded encoders matching original training data
    _city_enc = LabelEncoder()
    _city_enc.fit(['Ahmedabad', 'Bangalore', 'Chennai', 'Delhi', 'Hyderabad',
                   'Jaipur', 'Kolkata', 'Lucknow', 'Mumbai', 'Pune'])
    _weather_enc = LabelEncoder()
    _weather_enc.fit(['Clear', 'Fog', 'Rain'])
    _time_enc = LabelEncoder()
    _time_enc.fit(['Afternoon', 'Evening', 'Morning', 'Night'])
    _day_enc = LabelEncoder()
    _day_enc.fit(['Weekday', 'Weekend'])
    _road_enc = LabelEncoder()
    _road_enc.fit(['Highway', 'Main Road', 'Street'])

    _traffic_cat_encoders = {
        'City': _city_enc,
        'Weather': _weather_enc,
        'Time_of_Day': _time_enc,
        'Day_Type': _day_enc,
        'Road_Type': _road_enc,
    }

    df = pd.DataFrame([data])

    for col, enc in _traffic_cat_encoders.items():
        if col in df.columns:
            try:
                df[col] = enc.transform(df[col])
            except Exception:
                df[col] = enc.transform([enc.classes_[0]])[0]

    # Ensure correct feature order and numeric types
    if traffic_features:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df = df[traffic_features]
    else:
        df = df.select_dtypes(include=[np.number])

    return df

def encode_crime_data(data):
    """Encode categorical data for crime model"""
    df = pd.DataFrame([data])
    
    # Encode categorical columns if encoders exist
    if crime_encoders:
        for col in crime_encoders:
            if col in df.columns:
                try:
                    df[col] = crime_encoders[col].transform(df[col])
                except:
                    df[col] = crime_encoders[col].transform([crime_encoders[col].classes_[0]])[0]
    
    # Convert all to numeric
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/')
def home():
    """Health check endpoint"""
    status = {
        'citizen_model': citizen_model is not None,
        'house_model': house_model is not None,
        'traffic_model': traffic_model is not None,
        'crime_model': crime_model is not None
    }
    return jsonify({
        'message': '🚀 SMART CITY AI BACKEND IS RUNNING!',
        'status': 'online',
        'models': status,
        'endpoints': {
            '/predict/citizen': 'POST - Predict citizen segment',
            '/predict/house_price': 'POST - Predict house price',
            '/predict/traffic': 'POST - Predict traffic status',
            '/predict/crime': 'POST - Predict crime risk'
        }
    })

# 1. Citizen Segmentation
@app.route('/predict/citizen', methods=['POST'])
def predict_citizen():
    try:
        if citizen_model is None:
            return jsonify({'success': False, 'error': 'Citizen model not loaded'}), 500
        
        data = request.get_json()
        print(f"📊 Citizen Prediction Input: {data}")
        
        input_data = encode_citizen_data(data)
        prediction = citizen_model.predict(input_data)[0]
        
        return jsonify({
            'success': True,
            'citizen_segment': int(prediction),
            'segment_label': 'High Value Citizen' if prediction == 1 else 'Low Value Citizen'
        })
    except Exception as e:
        print(f"❌ Citizen Error: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 400

# 2. House Price Prediction
@app.route('/predict/house_price', methods=['POST'])
def predict_house_price():
    try:
        if house_model is None:
            return jsonify({'success': False, 'error': 'House price model not loaded'}), 500
        
        data = request.get_json()
        print(f"🏠 House Price Input: {data}")
        
        # Create DataFrame with correct feature order
        if house_features:
            input_data = pd.DataFrame([[
                float(data.get('Square_Footage', 0)),
                float(data.get('Num_Bathrooms', 0)),
                float(data.get('Num_Bedrooms', 0)),
                float(data.get('Lot_Size', 0)),
                float(data.get('Neighborhood_Quality', 0))
            ]], columns=house_features)
        else:
            input_data = pd.DataFrame([[
                float(data.get('Square_Footage', 0)),
                float(data.get('Num_Bathrooms', 0)),
                float(data.get('Num_Bedrooms', 0)),
                float(data.get('Lot_Size', 0)),
                float(data.get('Neighborhood_Quality', 0))
            ]])
        
        prediction = house_model.predict(input_data)[0]
        
        return jsonify({
            'success': True,
            'predicted_price': round(float(prediction), 2),
            'currency': 'USD'
        })
    except Exception as e:
        print(f"❌ House Price Error: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 400

# 3. Traffic Prediction
@app.route('/predict/traffic', methods=['POST'])
def predict_traffic():
    try:
        if traffic_model is None or traffic_scaler is None:
            return jsonify({'success': False, 'error': 'Traffic model not loaded'}), 500
        
        data = request.get_json()
        print(f"🚦 Traffic Input: {data}")
        
        # Prepare input data
        input_data = encode_traffic_data(data)
        
        # Scale features
        input_scaled = traffic_scaler.transform(input_data)
        
        # Predict
        prediction = traffic_model.predict(input_scaled)[0]
        
        return jsonify({
            'success': True,
            'traffic_high': int(prediction),
            'traffic_status': '🔴 HIGH TRAFFIC' if prediction == 1 else '🟢 LOW TRAFFIC'
        })
    except Exception as e:
        print(f"❌ Traffic Error: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 400

# 4. Crime Risk Prediction
@app.route('/predict/crime', methods=['POST'])
def predict_crime():
    try:
        if crime_model is None or crime_scaler is None:
            return jsonify({'success': False, 'error': 'Crime model not loaded'}), 500
        
        data = request.get_json()
        print(f"🚨 Crime Input: {data}")
        
        # Prepare input data
        input_data = encode_crime_data(data)
        
        # Scale features
        input_scaled = crime_scaler.transform(input_data)
        
        # Predict
        prediction = crime_model.predict(input_scaled)[0]
        
        return jsonify({
            'success': True,
            'crime_risk': int(prediction),
            'risk_level': '🔴 HIGH RISK' if prediction == 1 else '🟢 LOW RISK'
        })
    except Exception as e:
        print(f"❌ Crime Error: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 400

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# ============================================
# MAIN
# ============================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🌟 SMART CITY AI BACKEND READY!")
    print("📍 Server running at: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
