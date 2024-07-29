from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler  # Add this import

app = Flask(__name__)

test = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template("t.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the features from the form
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]

    # You need to apply scaling to the features here
    scaler = StandardScaler()  # Initialize StandardScaler
    final_scaled = scaler.fit_transform(final)  # Scale features

    # Predict using the loaded model
    prediction = test.predict(final_scaled)

    if prediction == 0:
        result = "Slight"
    elif prediction == 1:
        result = "Serious"
    else:
        result = "Fatal"

    return render_template('t.html', pred=f"\t\t\t\t\tProbability of accident severity is: {result}")

@app.route('/accuracy', methods=['GET'])
def accuracy():
    rf_accuracy = 0.92  # Replace with actual value
    gb_accuracy = 0.90  # Replace with actual value
    xgb_accuracy = 0.93  # Replace with actual value
    voting_accuracy = 0.94  # Replace with actual value

    feature_importances = {
        "Sex_Of_Driver": 0.1,
        "Vehicle_Type": 0.15,
        "Speed_limit": 0.25,
        "Road_Type": 0.2,
        "Number_of_Pasengers": 0.05,
        "Day_of_Week": 0.15,
        "Light_Conditions": 0.1
    }  # Replace with actual values

    return render_template('accuracy.html', rf_accuracy=rf_accuracy, gb_accuracy=gb_accuracy, xgb_accuracy=xgb_accuracy, voting_accuracy=voting_accuracy, feature_importances=feature_importances)

if __name__ == '__main__':
    app.run(debug=True)
