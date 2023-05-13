from flask import Flask, jsonify, request, session
from roboflow import Roboflow
import os

app = Flask(__name__)
# Initialize the Roboflow instance and set the API key
rf = Roboflow(api_key="T3mYNN9ubGhQnuAOzDfR")
project = rf.workspace().project("skin-burn-detiction")
model_version = 3
model = project.version(model_version).model


@app.route("/predict", methods=['POST'])
def predict():
    # Get the image_path parameter from the request header
    image_path = request.args.get('image_path')
    print(image_path)
    return model.predict(image_path, confidence=40, overlap=30).json()


if __name__ == '__main__':
    app.run(debug=True)
