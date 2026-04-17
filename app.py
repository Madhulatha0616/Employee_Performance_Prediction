from flask import Flask, render_template, request
import pandas as pd
import pickle

# Create Flask app
app = Flask(__name__)

# Load trained model and data
model = pickle.load(open("employee_model.pkl", "rb"))
data = pd.read_csv("employee.csv")

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    emp_input = request.form["emp"]

    # Find employee
    if emp_input.isdigit():
        emp = data[data["Employee_ID"] == int(emp_input)]
    else:
        emp = data[data["Name"].str.lower() == emp_input.lower()]

    if emp.empty:
        return render_template(
            "index.html",
            prediction_text="Employee not found"
        )

    # Predict performance
    features = emp[['Age','Experience','Job_Satisfaction',
                     'Work_Hours','Training_Hours']]
    prediction = model.predict(features)[0]

    # Outcomes logic
    if prediction == "Bad":
        score = 25
        risk = "High Risk"
        recommendation = "Immediate training required"
    elif prediction == "Average":
        score = 70
        risk = "Medium Risk"
        recommendation = "Skill improvement needed"
    elif prediction == "Good":
        score = 90
        risk = "Low Risk"
        recommendation = "Eligible for incentives"
    else:
        score = 100
        risk = "Very Low Risk"
        recommendation = "Promotion recommended"

    # Experience level
    exp = int(emp["Experience"].values[0])
    if exp <= 2:
        exp_level = "Fresher"
    elif exp <= 5:
        exp_level = "Mid-Level Employee"
    else:
        exp_level = "Senior Employee"

    return render_template(
        "index.html",
        prediction_text=f"Performance Prediction: {prediction}",
        score_text=f"Performance Score: {score}%",
        risk_text=f"Risk Level: {risk}",
        recommendation_text=f"HR Recommendation: {recommendation}",
        experience_text=f"Experience Level: {exp_level}"
    )

# Run app
if __name__ == "__main__":
    app.run(debug=True)
