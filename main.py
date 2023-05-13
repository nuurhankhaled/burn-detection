from flask import Flask, jsonify, request, session
from roboflow import Roboflow
import os
app = Flask(__name__)
# Initialize the Roboflow instance and set the API key
@app.route("/predict", methods=['POST'])
def predict():
    rf = Roboflow(api_key="T3mYNN9ubGhQnuAOzDfR")
    project = rf.workspace().project("skin-burn-detiction")
    model_version = 3
    model = project.version(model_version).model
    # Get the image URL from the query parameter
    image_url = request.args.get('image_url')
    print(image_url)
    return model.predict_from_url(image_url, confidence=40, overlap=30).json()


if __name__ == '__main__':
    app.run(debug=True)
