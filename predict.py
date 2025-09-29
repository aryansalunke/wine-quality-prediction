import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

def predict_wine_quality(data):
    # Load the trained model and scaler
    model = joblib.load('wine_quality_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Perform the same preprocessing as during training (scaling)
    # Assuming `data` is a DataFrame containing the features to predict on
    data_scaled = scaler.transform(data)

    # Make predictions
    predictions = model.predict(data_scaled)

    return predictions

# Example usage:
if __name__ == "__main__":
    # Sample data for prediction (replace with actual input)
    input_data = {
        'type': [1],  # encoded value for 'white' wine (replace with actual data)
        'fixed acidity': [7.4],
        'volatile acidity': [0.7],
        'citric acid': [0.0],
        'residual sugar': [1.9],
        'chlorides': [0.076],
        'free sulfur dioxide': [11.0],
        'total sulfur dioxide': [34.0],
        'density': [0.9978],
        'pH': [3.51],
        'sulphates': [0.56],
        'alcohol': [9.4]
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data)

    # Make predictions
    predictions = predict_wine_quality(input_df)
    print(f"Predicted Wine Quality: {predictions[0]}")
