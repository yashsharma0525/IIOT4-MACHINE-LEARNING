from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import traceback

app = Flask(__name__)
CORS(app)

# ============================================
# MODEL PATHS
# ============================================
MODEL_PATH = 'models'
CITIZEN_MODEL_PATH  = os.path.join(MODEL_PATH, 'citizen_segment_model.pkl')
CITIZEN_ENCODER_PATH= os.path.join(MODEL_PATH, 'label_encoders.pkl')
HOUSE_MODEL_PATH    = os.path.join(MODEL_PATH, 'house_price_model.pkl')
HOUSE_FEATURES_PATH = os.path.join(MODEL_PATH, 'house_price_features.pkl')
TRAFFIC_MODEL_PATH  = os.path.join(MODEL_PATH, 'traffic_model.pkl')
TRAFFIC_SCALER_PATH = os.path.join(MODEL_PATH, 'traffic_scaler.pkl')
TRAFFIC_FEATURES_PATH = os.path.join(MODEL_PATH, 'traffic_features.pkl')
TRAFFIC_ENCODER_PATH= os.path.join(MODEL_PATH, 'traffic_label_encoders.pkl')
CRIME_MODEL_PATH    = os.path.join(MODEL_PATH, 'crime_model.pkl')
CRIME_SCALER_PATH   = os.path.join(MODEL_PATH, 'crime_scaler.pkl')
CRIME_ENCODER_PATH  = os.path.join(MODEL_PATH, 'crime_label_encoders.pkl')

# ============================================
# DATASET PATHS  ← apne actual CSV paths yahan dalo
# ============================================
DATA_PATH = 'data'
HOUSE_CSV   = os.path.join(DATA_PATH, 'house_price.csv')
TRAFFIC_CSV = os.path.join(DATA_PATH, 'traffic.csv')
CRIME_CSV   = os.path.join(DATA_PATH, 'crime.csv')
CITIZEN_CSV = os.path.join(DATA_PATH, 'citizen.csv')

# ============================================
# LOAD MODELS
# ============================================
print("=" * 60)
print("🚀 SMART CITY AI BACKEND INITIALIZING...")
print("=" * 60)

try:
    citizen_model    = joblib.load(CITIZEN_MODEL_PATH)
    citizen_encoders = joblib.load(CITIZEN_ENCODER_PATH)
    print("✅ Citizen Segmentation Model Loaded")
except Exception as e:
    print(f"❌ Citizen Model Error: {e}")
    citizen_model = None; citizen_encoders = None

try:
    house_model = joblib.load(HOUSE_MODEL_PATH)
    try:    house_features = joblib.load(HOUSE_FEATURES_PATH)
    except: house_features = ['Square_Footage','Num_Bathrooms','Num_Bedrooms','Lot_Size','Neighborhood_Quality']
    print("✅ House Price Model Loaded")
except Exception as e:
    print(f"❌ House Price Error: {e}")
    house_model = None; house_features = None

try:
    traffic_model    = joblib.load(TRAFFIC_MODEL_PATH)
    traffic_scaler   = joblib.load(TRAFFIC_SCALER_PATH)
    traffic_features = joblib.load(TRAFFIC_FEATURES_PATH)
    try:    traffic_encoders = joblib.load(TRAFFIC_ENCODER_PATH)
    except: traffic_encoders = None
    print("✅ Traffic Model Loaded")
except Exception as e:
    print(f"❌ Traffic Model Error: {e}")
    traffic_model = None; traffic_scaler = None; traffic_features = None; traffic_encoders = None

try:
    crime_model  = joblib.load(CRIME_MODEL_PATH)
    crime_scaler = joblib.load(CRIME_SCALER_PATH)
    try:    crime_encoders = joblib.load(CRIME_ENCODER_PATH)
    except: crime_encoders = None
    print("✅ Crime Model Loaded")
except Exception as e:
    print(f"❌ Crime Model Error: {e}")
    crime_model = None; crime_scaler = None; crime_encoders = None

# ============================================
# LOAD DATASETS (for chart stats)
# ============================================
def load_csv(path):
    try:
        df = pd.read_csv(path)
        print(f"✅ Dataset loaded: {path} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"⚠️  Dataset not found: {path} — {e}")
        return None

df_house   = load_csv(HOUSE_CSV)
df_traffic = load_csv(TRAFFIC_CSV)
df_crime   = load_csv(CRIME_CSV)
df_citizen = load_csv(CITIZEN_CSV)

print("=" * 60)
print("🎯 INITIALIZATION COMPLETE")
print("=" * 60)

# ============================================
# HELPER FUNCTIONS
# ============================================
def encode_citizen_data(data):
    df = pd.DataFrame([data])
    for col in citizen_encoders:
        if col in df.columns:
            try:    df[col] = citizen_encoders[col].transform(df[col])
            except: df[col] = citizen_encoders[col].transform([df[col].iloc[0]])[0]
    return df

def encode_traffic_data(data):
    from sklearn.preprocessing import LabelEncoder
    _city_enc = LabelEncoder(); _city_enc.fit(['Ahmedabad','Bangalore','Chennai','Delhi','Hyderabad','Jaipur','Kolkata','Lucknow','Mumbai','Pune'])
    _weather_enc = LabelEncoder(); _weather_enc.fit(['Clear','Fog','Rain'])
    _time_enc = LabelEncoder(); _time_enc.fit(['Afternoon','Evening','Morning','Night'])
    _day_enc = LabelEncoder(); _day_enc.fit(['Weekday','Weekend'])
    _road_enc = LabelEncoder(); _road_enc.fit(['Highway','Main Road','Street'])
    _cat = {'City':_city_enc,'Weather':_weather_enc,'Time_of_Day':_time_enc,'Day_Type':_day_enc,'Road_Type':_road_enc}
    df = pd.DataFrame([data])
    for col, enc in _cat.items():
        if col in df.columns:
            try:    df[col] = enc.transform(df[col])
            except: df[col] = enc.transform([enc.classes_[0]])[0]
    if traffic_features:
        for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df = df[traffic_features]
    else:
        df = df.select_dtypes(include=[np.number])
    return df

def encode_crime_data(data):
    df = pd.DataFrame([data])
    if crime_encoders:
        for col in crime_encoders:
            if col in df.columns:
                try:    df[col] = crime_encoders[col].transform(df[col])
                except: df[col] = crime_encoders[col].transform([crime_encoders[col].classes_[0]])[0]
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df

# ============================================
# PREDICTION ENDPOINTS
# ============================================
@app.route('/')
def home():
    status = {
        'citizen_model': citizen_model is not None,
        'house_model':   house_model   is not None,
        'traffic_model': traffic_model is not None,
        'crime_model':   crime_model   is not None,
        'house_dataset':   df_house   is not None,
        'traffic_dataset': df_traffic is not None,
        'crime_dataset':   df_crime   is not None,
        'citizen_dataset': df_citizen is not None,
    }
    return jsonify({'message':'🚀 SMART CITY AI BACKEND IS RUNNING!','status':'online','models':status})

@app.route('/predict/citizen', methods=['POST'])
def predict_citizen():
    try:
        if citizen_model is None: return jsonify({'success':False,'error':'Citizen model not loaded'}),500
        data = request.get_json()
        input_data = encode_citizen_data(data)
        prediction = citizen_model.predict(input_data)[0]
        return jsonify({'success':True,'citizen_segment':int(prediction),'segment_label':'High Value Citizen' if prediction==1 else 'Low Value Citizen'})
    except Exception as e:
        print(traceback.format_exc()); return jsonify({'success':False,'error':str(e)}),400

@app.route('/predict/house_price', methods=['POST'])
def predict_house_price():
    try:
        if house_model is None: return jsonify({'success':False,'error':'House price model not loaded'}),500
        data = request.get_json()
        cols = house_features or ['Square_Footage','Num_Bathrooms','Num_Bedrooms','Lot_Size','Neighborhood_Quality']
        input_data = pd.DataFrame([[float(data.get('Square_Footage',0)),float(data.get('Num_Bathrooms',0)),float(data.get('Num_Bedrooms',0)),float(data.get('Lot_Size',0)),float(data.get('Neighborhood_Quality',0))]],columns=cols)
        prediction = house_model.predict(input_data)[0]
        return jsonify({'success':True,'predicted_price':round(float(prediction),2),'currency':'USD'})
    except Exception as e:
        print(traceback.format_exc()); return jsonify({'success':False,'error':str(e)}),400

@app.route('/predict/traffic', methods=['POST'])
def predict_traffic():
    try:
        if traffic_model is None or traffic_scaler is None: return jsonify({'success':False,'error':'Traffic model not loaded'}),500
        data = request.get_json()
        input_data = encode_traffic_data(data)
        input_scaled = traffic_scaler.transform(input_data)
        prediction = traffic_model.predict(input_scaled)[0]
        return jsonify({'success':True,'traffic_high':int(prediction),'traffic_status':'🔴 HIGH TRAFFIC' if prediction==1 else '🟢 LOW TRAFFIC'})
    except Exception as e:
        print(traceback.format_exc()); return jsonify({'success':False,'error':str(e)}),400

@app.route('/predict/crime', methods=['POST'])
def predict_crime():
    try:
        if crime_model is None or crime_scaler is None: return jsonify({'success':False,'error':'Crime model not loaded'}),500
        data = request.get_json()
        input_data = encode_crime_data(data)
        input_scaled = crime_scaler.transform(input_data)
        prediction = crime_model.predict(input_scaled)[0]
        return jsonify({'success':True,'crime_risk':int(prediction),'risk_level':'🔴 HIGH RISK' if prediction==1 else '🟢 LOW RISK'})
    except Exception as e:
        print(traceback.format_exc()); return jsonify({'success':False,'error':str(e)}),400

# ============================================
# CHART STATS ENDPOINTS  ← NEW
# ============================================

@app.route('/stats/house', methods=['POST'])
def stats_house():
    """
    Returns chart data for house price.
    POST body (optional): { "Square_Footage": 2500, "Neighborhood_Quality": 7 }
    These are the currently selected row values — used to highlight the right bucket.
    """
    try:
        data = request.get_json(silent=True) or {}
        sel_sqft = float(data.get('Square_Footage', -1))
        sel_nq   = int(data.get('Neighborhood_Quality', -1))

        if df_house is not None:
            df = df_house.copy()
        else:
            return jsonify({'success': False, 'error': 'House dataset not loaded'}), 500

        # --- Bar: Avg Price per Sqft bucket ---
        def sqft_bucket(sq):
            sq = float(sq)
            if sq < 1500:  return '<1500'
            if sq < 2500:  return '1500-2500'
            if sq < 3500:  return '2500-3500'
            return '>3500'

        df['_bucket'] = df['Square_Footage'].apply(sqft_bucket)
        bar_data = df.groupby('_bucket')['House_Price'].mean().reindex(['<1500','1500-2500','2500-3500','>3500']).fillna(0)
        sel_bucket = sqft_bucket(sel_sqft) if sel_sqft >= 0 else None

        # --- Pie: Neighborhood Quality distribution ---
        def nq_label(q):
            q = int(q)
            if q <= 3: return 'Low'
            if q <= 7: return 'Medium'
            return 'High'

        df['_nq'] = df['Neighborhood_Quality'].apply(nq_label)
        pie_data = df['_nq'].value_counts().reindex(['Low','Medium','High']).fillna(0)
        sel_nq_label = nq_label(sel_nq) if sel_nq >= 0 else None

        return jsonify({
            'success': True,
            'bar': {
                'labels': list(bar_data.index),
                'values': [round(v/1000, 1) for v in bar_data.values],
                'selected': sel_bucket
            },
            'pie': {
                'labels': list(pie_data.index),
                'values': [int(v) for v in pie_data.values],
                'selected': sel_nq_label
            }
        })
    except Exception as e:
        print(traceback.format_exc()); return jsonify({'success':False,'error':str(e)}),400


@app.route('/stats/traffic', methods=['POST'])
def stats_traffic():
    """
    POST body (optional): { "City": "Mumbai", "Weather": "Rain" }
    """
    try:
        data = request.get_json(silent=True) or {}
        sel_city    = data.get('City', None)
        sel_weather = data.get('Weather', None)

        if df_traffic is not None:
            df = df_traffic.copy()
        else:
            return jsonify({'success': False, 'error': 'Traffic dataset not loaded'}), 500

        # --- Bar: High / Low per city ---
        city_grp = df.groupby('City')['Traffic_High'].value_counts().unstack(fill_value=0)
        if 1 not in city_grp.columns: city_grp[1] = 0
        if 0 not in city_grp.columns: city_grp[0] = 0
        city_grp = city_grp.sort_index()
        cities = list(city_grp.index)

        # --- Pie: Weather distribution ---
        wx_counts = df['Weather'].value_counts().reindex(['Clear','Rain','Fog']).fillna(0)

        return jsonify({
            'success': True,
            'bar': {
                'labels':   cities,
                'high':     [int(city_grp.loc[c, 1]) for c in cities],
                'low':      [int(city_grp.loc[c, 0]) for c in cities],
                'selected': sel_city
            },
            'pie': {
                'labels':   list(wx_counts.index),
                'values':   [int(v) for v in wx_counts.values],
                'selected': sel_weather
            }
        })
    except Exception as e:
        print(traceback.format_exc()); return jsonify({'success':False,'error':str(e)}),400


@app.route('/stats/crime', methods=['POST'])
def stats_crime():
    """
    POST body (optional): { "City": "Delhi", "Weapon_Used": "Gun" }
    """
    try:
        data = request.get_json(silent=True) or {}
        sel_city   = data.get('City', None)
        sel_weapon = data.get('Weapon_Used', None)
        # normalise weapons to 4 buckets same as frontend
        wmap = {'None':'None','Gun':'Gun','Knife':'Knife','Rod':'Rod','Axe':'Rod','Pipe':'Rod','Bottle':'Rod'}
        sel_weapon_norm = wmap.get(sel_weapon, sel_weapon)

        if df_crime is not None:
            df = df_crime.copy()
        else:
            return jsonify({'success': False, 'error': 'Crime dataset not loaded'}), 500

        # --- Bar: crimes per city ---
        crime_city = df[df['Crime_Occurred']==1]['City'].value_counts().sort_index()
        cities = list(crime_city.index)

        # --- Pie: weapon distribution ---
        df['_weapon_norm'] = df['Weapon_Used'].map(lambda w: wmap.get(w, 'Rod'))
        wp_counts = df['_weapon_norm'].value_counts().reindex(['None','Gun','Knife','Rod']).fillna(0)

        return jsonify({
            'success': True,
            'bar': {
                'labels':   cities,
                'values':   [int(v) for v in crime_city.values],
                'selected': sel_city
            },
            'pie': {
                'labels':   list(wp_counts.index),
                'values':   [int(v) for v in wp_counts.values],
                'selected': sel_weapon_norm
            }
        })
    except Exception as e:
        print(traceback.format_exc()); return jsonify({'success':False,'error':str(e)}),400


@app.route('/stats/citizen', methods=['POST'])
def stats_citizen():
    """
    POST body (optional): { "Area_Type": "Urban", "Employment_Type": "Private" }
    """
    try:
        data = request.get_json(silent=True) or {}
        sel_area = data.get('Area_Type', None)
        sel_emp  = data.get('Employment_Type', None)

        if df_citizen is not None:
            df = df_citizen.copy()
        else:
            return jsonify({'success': False, 'error': 'Citizen dataset not loaded'}), 500

        # --- Bar: Segment 0/1 per Area_Type ---
        area_grp = df.groupby('Area_Type')['Citizen_Segment'].value_counts().unstack(fill_value=0)
        if 1 not in area_grp.columns: area_grp[1] = 0
        if 0 not in area_grp.columns: area_grp[0] = 0
        area_grp = area_grp.reindex(['Urban','Rural','Semi-Urban']).fillna(0)
        areas = list(area_grp.index)

        # --- Pie: Employment distribution ---
        emp_counts = df['Employment_Type'].value_counts().reindex(['Private','Government','Unemployed','Self-Employed']).fillna(0)

        return jsonify({
            'success': True,
            'bar': {
                'labels':   areas,
                'seg1':     [int(area_grp.loc[a, 1]) for a in areas],
                'seg0':     [int(area_grp.loc[a, 0]) for a in areas],
                'selected': sel_area
            },
            'pie': {
                'labels':   list(emp_counts.index),
                'values':   [int(v) for v in emp_counts.values],
                'selected': sel_emp
            }
        })
    except Exception as e:
        print(traceback.format_exc()); return jsonify({'success':False,'error':str(e)}),400


# ============================================
# ERROR HANDLERS
# ============================================
@app.errorhandler(404)
def not_found(e):    return jsonify({'success':False,'error':'Endpoint not found'}),404
@app.errorhandler(500)
def server_error(e): return jsonify({'success':False,'error':'Internal server error'}),500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌟 SMART CITY AI BACKEND READY!")
    print("📍 Server: http://localhost:5000")
    print("📊 New chart endpoints:")
    print("   POST /stats/house")
    print("   POST /stats/traffic")
    print("   POST /stats/crime")
    print("   POST /stats/citizen")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
