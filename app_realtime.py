from flask import Flask, render_template
from flask_socketio import SocketIO
import tensorflow as tf
import numpy as np
from keras.models import load_model
import json
import boto3

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

# AWS S3 configuration
AWS_ACCESS_KEY_ID = 'AKIAT2OAYY5QIW6BUNJU'
AWS_SECRET_ACCESS_KEY = 'Y8ScYVnY4VKs54AoLcRRKsvXbAID6MU8KqktjuON'
AWS_BUCKET_NAME = 'dummysliit'

s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

@app.route('/')
def realtime():
    return render_template('realtime.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetch data from S3
        s3_response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key='sensor_data.json')
        sensor_data = json.loads(s3_response['Body'].read().decode('utf-8'))

        # Extract input values from the fetched data
        temperature = float(sensor_data['temperature'])
        dew_point = float(sensor_data['dew_point'])
        humidity = float(sensor_data['humidity'])
        wind_speed = float(sensor_data['wind_speed'])
        pressure = float(sensor_data['pressure'])

        # Preprocess the input values
        inputs = np.array([[temperature, dew_point, humidity, wind_speed, pressure]])

        # Make prediction using the loaded model
        prediction = model.predict(inputs)
        predicted_class = np.argmax(prediction, axis=1)[0]
        weather_prediction = weather_conditions[predicted_class]

        # Send real-time data to connected clients
        socketio.emit('sensor_data', json.dumps(sensor_data))
        socketio.emit('weather_prediction', weather_prediction)  # Emit the prediction

        return render_template('realtime.html', prediction=weather_prediction)
    except Exception as e:
        return render_template('realtime.html', error=str(e))

@app.route('/get_sensor_data')
def get_sensor_data():
    try:
        # Fetch data from S3
        s3_response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key='sensor_data.json')
        sensor_data = json.loads(s3_response['Body'].read().decode('utf-8'))
        return json.dumps(sensor_data)
    except Exception as e:
        return json.dumps({'error': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True)
