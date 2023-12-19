from flask import Flask, request, jsonify
import joblib  # or any other library to load your model
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your trained model here
model = joblib.load('model_v2.h5')  # Update with the actual path to your model file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        preference = data['preference']

        # Make predictions using the loaded model
        prediction = model.predict(preference)
        print(prediction)

        return jsonify(prediction)

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/hello', methods=['GET'])
def hello_world():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(debug=True)
