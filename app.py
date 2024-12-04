from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Load the trained model and scaler
model = joblib.load('best_model_8_features.pkl')  # Adjusted for new model
scaler = joblib.load('scaler_8_features.pkl')  # Adjusted for new scaler

app = Flask(__name__)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:root@localhost/mobilepricedB'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define the Database Model
class MobilePrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    resoloution = db.Column(db.Float, nullable=False)
    ppi = db.Column(db.Float, nullable=False)
    cpu_core = db.Column(db.Integer, nullable=False)
    cpu_freq = db.Column(db.Float, nullable=False)
    ram = db.Column(db.Float, nullable=False)
    rear_cam = db.Column(db.Float, nullable=False)
    battery = db.Column(db.Integer, nullable=False)
    thickness = db.Column(db.Float, nullable=False)
    predicted_price = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the features from the frontend
        data = request.get_json()

        # Log the received data for debugging
        print("Received data:", data)

        # Ensure all expected features are provided and are valid
        required_features = [
            'resoloution', 'ppi', 'cpu_core', 'cpu_freq', 'ram', 'RearCam', 'battery', 'thickness'
        ]
        
        # Check if all required features are present
        for feature in required_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
        
        # Extract feature values (ensure they are valid numbers)
        features = np.array([[ 
            data['resoloution'],
            data['ppi'],
            data['cpu_core'],
            data['cpu_freq'],
            data['ram'],
            data['RearCam'],
            data['battery'],
            data['thickness']
        ]])

        # Validate if all feature values are valid (not NaN or None)
        if np.any(np.isnan(features)):
            print("Invalid feature values:", features)
            return jsonify({'error': 'Invalid feature value(s) detected (NaN).'}), 400

        # Scale the features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]  # Get the first value from the prediction array
        print("Predicted Price:", prediction)

        # Save the input data and prediction to the database
        new_entry = MobilePrediction(
            resoloution=data['resoloution'],
            ppi=data['ppi'],
            cpu_core=data['cpu_core'],
            cpu_freq=data['cpu_freq'],
            ram=data['ram'],
            rear_cam=data['RearCam'],
            battery=data['battery'],
            thickness=data['thickness'],
            predicted_price=float(prediction)
        )

        # Try to save to the database
        try:
            db.session.add(new_entry)
            db.session.commit()
            print("Data saved to DB successfully.")
        except Exception as e:
            print("Error while saving to DB:", e)
            return jsonify({'error': 'Error while saving to DB: ' + str(e)}), 500

        # Fetch the latest predicted price from the database
        latest_prediction = MobilePrediction.query.order_by(MobilePrediction.created_at.desc()).first()

        # Return the prediction from the database as JSON
        return jsonify({
            'predicted_price': latest_prediction.predicted_price
        })

    except Exception as e:
        print("Error:", e)  # Print the error message for debugging
        return jsonify({'error': str(e)}), 400

    try:
        # Get the features from the frontend
        data = request.get_json()

        # Print the received data for debugging
        print("Received data:", data)

        # Ensure all expected features are provided and are valid
        required_features = [
            'resoloution', 'ppi', 'cpu_core', 'cpu_freq', 'ram', 'RearCam', 'battery', 'thickness'
        ]
        
        # Check if all required features are present
        for feature in required_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
        
        # Extract feature values (ensure they are valid numbers)
        features = np.array([[ 
            data['resoloution'],
            data['ppi'],
            data['cpu_core'],
            data['cpu_freq'],
            data['ram'],
            data['RearCam'],
            data['battery'],
            data['thickness']
        ]])

        # Validate if all feature values are valid (not NaN or None)
        if np.any(np.isnan(features)):
            return jsonify({'error': 'Invalid feature value(s) detected (NaN).'}), 400

        # Scale the features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]  # Get the first value from the prediction array

        # Save the input data and prediction to the database
        new_entry = MobilePrediction(
            resoloution=data['resoloution'],
            ppi=data['ppi'],
            cpu_core=data['cpu_core'],
            cpu_freq=data['cpu_freq'],
            ram=data['ram'],
            rear_cam=data['RearCam'],
            battery=data['battery'],
            thickness=data['thickness'],
            predicted_price=float(prediction)
        )
        db.session.add(new_entry)
        db.session.commit()

        # Fetch the latest predicted price from the database
        latest_prediction = MobilePrediction.query.order_by(MobilePrediction.created_at.desc()).first()

        # Return the prediction from the database as JSON
        return jsonify({
            'predicted_price': latest_prediction.predicted_price
        })

    except Exception as e:
        print("Error:", e)  # Print the error message for debugging
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    with app.app_context():  # Activate the app context
        db.create_all()  # Create tables in the database if they don't exist
    app.run(debug=False)
