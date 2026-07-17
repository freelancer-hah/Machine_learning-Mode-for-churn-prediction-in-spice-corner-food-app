import os
import requests
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

NODE_BACKEND_URL = os.getenv('NODE_BACKEND_URL', 'http://localhost:5000')
PORT = int(os.getenv('PORT', 5001))
HOST = os.getenv('HOST', '0.0.0.0')
MODEL_RETRAIN_INTERVAL = int(os.getenv('MODEL_RETRAIN_INTERVAL', 604800))
MIN_TRAINING_SAMPLES = int(os.getenv('MIN_TRAINING_SAMPLES', 50))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# ============================================
# LOGGING
# ============================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info(f"✅ Backend URL: {NODE_BACKEND_URL}")
logger.info(f"✅ Port: {PORT}")
logger.info(f"✅ Host: {HOST}")

# ============================================
# FLASK APP
# ============================================

app = Flask(__name__)
CORS(app)

# ============================================
# MODEL PATHS
# ============================================

MODEL_PATH = 'models/churn_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
ENCODER_PATH = 'models/encoder.pkl'
FEATURES_PATH = 'models/features.json'

FEATURES = [
    'days_since_last_order',
    'total_orders',
    'avg_order_value',
    'total_spent',
    'avg_items_per_order',
    'weekend_orders_ratio',
    'night_orders_ratio',
    'recent_orders_3months',
    'previous_orders_3months',
    'order_frequency_days'
]

model = None
scaler = None
label_encoder = None

# ============================================
# MODEL FUNCTIONS
# ============================================

def load_or_train_model():
    """Load existing model or train new one"""
    global model, scaler, label_encoder
    
    os.makedirs('models', exist_ok=True)
    logger.info("📁 models/ folder ready")
    
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        logger.info("✅ Loading existing model...")
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            with open(ENCODER_PATH, 'rb') as f:
                label_encoder = pickle.load(f)
            logger.info("✅ Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
    
    logger.info("🔄 No model found. Training new model...")
    return train_model()

def train_model():
    """Train ML model on customer data"""
    global model, scaler, label_encoder
    
    try:
        logger.info(f"📡 Fetching data from: {NODE_BACKEND_URL}/api/churn/training-data")
        response = requests.get(
            f'{NODE_BACKEND_URL}/api/churn/training-data',
            timeout=30
        )
        if response.status_code != 200:
            logger.error(f"❌ API error: {response.status_code}")
            data = generate_synthetic_data()
        else:
            data = response.json()
            logger.info(f"✅ Fetched {len(data)} customer records")
    except Exception as e:
        logger.error(f"❌ Error fetching data: {e}")
        data = generate_synthetic_data()
    
    if len(data) < MIN_TRAINING_SAMPLES:
        logger.warning(f"⚠️ Not enough data, using synthetic data")
        data = generate_synthetic_data()
    
    df = pd.DataFrame(data)
    df = engineer_features(df)
    
    label_encoder = LabelEncoder()
    df['preferred_category'] = label_encoder.fit_transform(df['preferred_category'])
    
    X = df[FEATURES]
    y = df['churned']
    X = X.fillna(X.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    logger.info(f"🎯 Model Accuracy: {accuracy:.2%}")
    logger.info(f"📊 ROC-AUC: {auc:.2%}")
    
    os.makedirs('models', exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    with open(FEATURES_PATH, 'w') as f:
        json.dump(FEATURES, f)
    
    logger.info("✅ Model saved successfully")
    return True

def engineer_features(df):
    """Create new features"""
    df['weekend_orders_ratio'] = df['weekend_orders'] / (df['total_orders'] + 1)
    df['night_orders_ratio'] = df['night_orders'] / (df['total_orders'] + 1)
    return df

def generate_synthetic_data():
    """Generate synthetic data for testing"""
    logger.info("🔄 Generating synthetic data...")
    np.random.seed(42)
    data = []
    for i in range(100):
        total_orders = np.random.randint(1, 20)
        days_since = np.random.randint(1, 60)
        data.append({
            'customer_id': f'cust_{i}',
            'days_since_last_order': days_since,
            'total_orders': total_orders,
            'avg_order_value': np.random.randint(200, 1500),
            'total_spent': np.random.randint(500, 20000),
            'total_items': np.random.randint(2, 30),
            'avg_items_per_order': np.random.randint(1, 5),
            'preferred_category': np.random.choice(['Main Course', 'Fast Food', 'Starters', 'Beverages', 'Desserts']),
            'weekend_orders': np.random.randint(0, 10),
            'night_orders': np.random.randint(0, 10),
            'recent_orders_3months': np.random.randint(0, 8),
            'previous_orders_3months': np.random.randint(0, 8),
            'order_frequency_days': np.random.randint(5, 30),
            'churned': 1 if (days_since > 30 and total_orders < 5) else 0
        })
    return data

def get_risk_level(probability):
    if probability > 0.7:
        return 'High'
    elif probability > 0.4:
        return 'Medium'
    else:
        return 'Low'

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'backend_url': NODE_BACKEND_URL,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        customer_data = request.json
        if model is None:
            load_or_train_model()
        
        df = pd.DataFrame([customer_data])
        df = engineer_features(df)
        df['preferred_category'] = label_encoder.transform(df['preferred_category'])
        
        X = df[FEATURES].fillna(df[FEATURES].median())
        X_scaled = scaler.transform(X)
        
        probability = model.predict_proba(X_scaled)[0][1]
        prediction = model.predict(X_scaled)[0]
        
        return jsonify({
            'customer_id': customer_data.get('customer_id'),
            'churn_probability': float(probability),
            'churn_risk': float(probability * 100),
            'prediction': 'At Risk' if prediction == 1 else 'Active',
            'risk_level': get_risk_level(probability),
            'confidence': float(max(probability, 1 - probability))
        })
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    try:
        customers = request.json.get('customers', [])
        if model is None:
            load_or_train_model()
        
        results = []
        for customer in customers:
            df = pd.DataFrame([customer])
            df = engineer_features(df)
            df['preferred_category'] = label_encoder.transform(df['preferred_category'])
            
            X = df[FEATURES].fillna(df[FEATURES].median())
            X_scaled = scaler.transform(X)
            
            probability = model.predict_proba(X_scaled)[0][1]
            prediction = model.predict(X_scaled)[0]
            
            results.append({
                'customer_id': customer.get('customer_id'),
                'churn_probability': float(probability),
                'churn_risk': float(probability * 100),
                'prediction': 'At Risk' if prediction == 1 else 'Active',
                'risk_level': get_risk_level(probability)
            })
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        success = train_model()
        return jsonify({
            'success': success,
            'message': 'Model retrained successfully' if success else 'Retraining failed'
        })
    except Exception as e:
        logger.error(f"❌ Retrain error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/features', methods=['GET'])
def get_features():
    return jsonify({
        'features': FEATURES,
        'feature_importance': dict(zip(FEATURES, model.feature_importances_)) if model else {}
    })

if __name__ == '__main__':
    load_or_train_model()
    app.run(host=HOST, port=PORT)