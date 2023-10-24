from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Create a dictionary to store information about each cancer stage
stage_info = {
    'Normal': {
        'name': 'Normal',
        'details': 'This is the normal stage, which indicates no signs of cancer.',
        'tip1': 'Regular check-ups are important for maintaining your health.',
         "tip2": "Maintain a healthy lifestyle with regular exercise and a balanced diet.",
        "tip3": "Don't smoke and avoid exposure to harmful pollutants.",
        "tip4": "Get regular check-ups and screenings for early detection."
    },
    'Stage1': {
        'name': 'Stage 1',
        'details': 'Stage 1 cancer indicates an early stage of cancer development.',
        'tip1': 'Consult with your doctor for further evaluation and treatment options.',
          "tip2": "Follow the recommended treatment plan from your healthcare provider.",
        "tip3": "Consider seeking a second opinion for your treatment options.",
        "tip4": "Lean on your support system of friends and family for emotional support."
    },
    'Stage2': {
        'name': 'Stage 2',
        'details': 'Stage 2 cancer suggests a moderately advanced stage of cancer.',
        'tip1': 'Early detection and treatment are crucial at this stage.',
         "tip2": "Work closely with your healthcare provider to discuss the best treatment options for stage 2.",
        "tip3": "Continue to follow up with your healthcare team regularly.",
        "tip4": "Consider joining a support group for additional emotional support."
    },
    'Stage3': {
        'name': 'Stage 3',
        'details': 'Stage 3 cancer indicates an advanced stage of cancer that may have spread to nearby tissues.',
        'tip1': 'Seek immediate medical attention and consult with specialists.',
         "tip2": "Consult with your healthcare team to explore advanced treatment options.",
        "tip3": "Consider joining a support group to connect with others going through a similar situation.",
        "tip4": "Maintain a positive outlook and focus on your well-being."
    }
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Load the trained model inside the route function
    model = load_model("model/model_weights.hdf5")
    
    # Define the labels
    labels = ['Normal', 'Stage1', 'Stage2', 'Stage3']

    if request.method == "POST":
        uploaded_image = request.files.get("image")

        if uploaded_image:
            image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
            image = cv2.resize(image, (32, 32))
            image = image.reshape(1, 32, 32, 3)
            image = image.astype('float32') / 255
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction)

            # Get the name, details, and tips for the predicted stage
            predicted_stage = labels[predicted_class]
            stage_name = stage_info[predicted_stage]['name']
            stage_details = stage_info[predicted_stage]['details']
            stage_tips = [stage_info[predicted_stage]['tip1'],
                          stage_info[predicted_stage]['tip2'],
                          stage_info[predicted_stage]['tip3'],
                          stage_info[predicted_stage]['tip4']]

            # Pass the prediction result and additional stage information to the template
            return render_template("index.html", prediction=stage_name, details=stage_details, tips=stage_tips)

    # Render the template without a prediction result initially
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
