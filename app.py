from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load Model
model = pickle.load(open("diabetes_model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Create DataFrame
    input_data = pd.DataFrame({
        "Pregnancies": [float(request.form["Pregnancies"])],
        "Glucose": [float(request.form["Glucose"])],
        "BloodPressure": [float(request.form["BloodPressure"])],
        "SkinThickness": [float(request.form["SkinThickness"])],
        "Insulin": [float(request.form["Insulin"])],
        "BMI": [float(request.form["BMI"])],
        "DiabetesPedigreeFunction": [float(request.form["DiabetesPedigreeFunction"])],
        "Age": [float(request.form["Age"])]
    })

    # Prediction
    prediction = model.predict(input_data)

    # Probability
    probability = model.predict_proba(input_data)
    risk = round(probability[0][1] * 100, 2)

    # Result
    if prediction[0] == 1:
        result = f"⚠️ Diabetic | Risk Score: {risk}%"
    else:
        result = f"✅ Not Diabetic | Risk Score: {risk}%"

    return render_template(
        "index.html",
        prediction_text=result,

        Pregnancies=request.form["Pregnancies"],
        Glucose=request.form["Glucose"],
        BloodPressure=request.form["BloodPressure"],
        SkinThickness=request.form["SkinThickness"],
        Insulin=request.form["Insulin"],
        BMI=request.form["BMI"],
        DiabetesPedigreeFunction=request.form["DiabetesPedigreeFunction"],
        Age=request.form["Age"]
    )


if __name__ == "__main__":
    app.run(debug=True)