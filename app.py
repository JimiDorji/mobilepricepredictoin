from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
# Load the trained model and scaler
model = joblib.load('best_model_8_features.pkl')  # Adjusted for new model
scaler = joblib.load('scaler_8_features.pkl')  # Adjusted for new scaler
# Define the path to the 'data' folder inside the static folder
IMAGE_FOLDER = os.path.join('static', 'images')

app = Flask(__name__)

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

        # Return the prediction as JSON
        return jsonify({
            'predicted_price': prediction
        })

    except Exception as e:
        print("Error:", e)  # Print the error message for debugging
        return jsonify({'error': str(e)}), 400
    
@app.route('/data-analysis')
def data_analysis():
    # List all image files in the 'images' folder inside 'static'
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png'))]
    return render_template('data-analysis.html', images=image_files)



if __name__ == '__main__':
    app.run(debug=True)
