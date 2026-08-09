from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Define paths to the model and scaler files
model_path = 'wine_quality_lgb_model.pkl'  # Replace with the model that was saved (XGBoost or LightGBM)
scaler_path = 'scaler.pkl'  # Path to the saved scaler

# Load the trained model and scaler
try:
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)  # Load the model (XGBoost or LightGBM)
        scaler = joblib.load(scaler_path)  # Load the scaler
        print("Model and scaler loaded successfully.")
    else:
        raise FileNotFoundError("Model or scaler file not found.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

@app.route('/')
def index():
    return render_template('index.html')  # Render the initial form (index.html)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler could not be loaded. Please check the server configuration.'}), 500

    try:
        # Extract features from the form
        features = [
            float(request.form['type']),  # 0 for white, 1 for red
            float(request.form['fixed_acidity']),
            float(request.form['volatile_acidity']),
            float(request.form['citric_acid']),
            float(request.form['residual_sugar']),
            float(request.form['chlorides']),
            float(request.form['free_sulfur_dioxide']),
            float(request.form['total_sulfur_dioxide']),
            float(request.form['density']),
            float(request.form['pH']),
            float(request.form['sulphates']),
            float(request.form['alcohol']),
        ]

        # Ensure the model expects the correct number of features
        if len(features) != model.n_features_in_:
            raise ValueError(f"Expected {model.n_features_in_} features, but received {len(features)}.")

        # Reshape features and scale them
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)
        predicted_quality = round(prediction[0], 2)

        # Return the prediction
        return render_template('index.html', prediction_text=f'Predicted Wine Quality: {predicted_quality}')

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)