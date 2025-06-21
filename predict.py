import joblib
import requests
import pandas as pd

# Configuration
WEATHER_API_KEY = "18a9e977d32e4a7a8e961308252106"
LOCATION = "Pune"
MODEL_PATH = "D:/IndustrialOvenHeatUpPrediction/oven_time_predictor.pkl"  # Path to your trained model

# Sensor targets
SENSOR_TARGETS = {
    'WU311': 160,
    'WU312': 190,
    'WU314': 190,
    'WU321': 190,
    'WU322': 190,
    'WU323': 190
}

def get_weather():
    """Fetch current weather data"""
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={LOCATION}"
    response = requests.get(url)
    data = response.json()
    return {
        "temp": data["current"]["temp_c"],
        "humidity": data["current"]["humidity"]
    }

def predict():
    """Make predictions using the trained model"""
    try:
        # Load model and feature names
        model, feature_names = joblib.load(MODEL_PATH)
        
        # Get user input
        current_temp = float(input("Enter current oven temperature (°C): "))
        sensor = input("Enter sensor (WU311/WU312/WU314/WU321/WU322/WU323): ").strip().upper()
        
        if sensor not in SENSOR_TARGETS:
            raise ValueError("Invalid sensor type")

        # Get weather
        weather = get_weather()

        # Prepare input data
        input_data = pd.DataFrame({
            'start_temp': [current_temp],
            'ambient_temp': [weather['temp']],
            'humidity': [weather['humidity']],
            'target_temp': [SENSOR_TARGETS[sensor]],
            'sensor_WU311': [1 if sensor == 'WU311' else 0],
            'sensor_WU312': [1 if sensor == 'WU312' else 0],
            'sensor_WU314': [1 if sensor == 'WU314' else 0],
            'sensor_WU321': [1 if sensor == 'WU321' else 0],
            'sensor_WU322': [1 if sensor == 'WU322' else 0],
            'sensor_WU323': [1 if sensor == 'WU323' else 0]
        })[feature_names]  # Ensure correct feature order

        # Predict
        prediction = model.predict(input_data)[0]
        
        print(f"\nPredicted time to target: {prediction:.1f} minutes")
        print(f"Current weather: {weather['temp']}°C, {weather['humidity']}% humidity")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    predict()