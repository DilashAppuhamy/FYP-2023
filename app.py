from flask import Flask, render_template, request
from flask_socketio import SocketIO
import tensorflow as tf
import numpy as np
from keras.models import load_model
import json

app = Flask(__name__)
socketio = SocketIO(app)

# Load your pre-trained model
model = tf.keras.models.load_model('E:\\Sliit\\4th yr 1st sem\\project\\2nd sem\\Weather prediction sensor data\\Untitled Folder\\actual_model\\model_new')

# Mapping of indices to weather conditions
weather_conditions = {
    0: "Fog",
    1: "Freezing",
    2: "Modtly Cloudy",
    3: "Cloudy",
    4: "Rain",
    5: "Rain Showers",
    6: "Mainly clear",
    7: "Snow showers",
    8: "Snow",
    9: "Clear"
    
    
    # Add more conditions as needed
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        temperature = float(request.form['temperature'])
        dew_point = float(request.form['dew_point'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])
        pressure = float(request.form['pressure'])

        # Preprocess the input values
        inputs = np.array([[temperature, dew_point, humidity, wind_speed, pressure]])
        
        # Make prediction using the loaded model
        prediction = model.predict(inputs)
        predicted_class = np.argmax(prediction, axis=1)[0]
        weather_prediction = weather_conditions[predicted_class]

          # Send real-time data to connected clients
        sensor_data = {
            'temperature': temperature,
            'dew_point': dew_point,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'pressure': pressure,
            'prediction': weather_prediction
        }
        socketio.emit('sensor_data', json.dumps(sensor_data))

        return render_template('index.html', prediction=weather_prediction)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
   # socketio.run(app, debug=True)
   app.run(debug=True)