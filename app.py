import numpy as np
from flask import Flask, render_template, request
import pickle

# Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


# default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')


# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2) * 100
    return render_template('index.html', prediction_text='The Chance Of Admittance is:{}%'.format(output))


# Run Server
if __name__ == "__main__":
    app.run(debug=True)
