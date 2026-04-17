import streamlit as st
import pandas as pd
import pickle

# Load model and data
model = pickle.load(open("employee_model.pkl", "rb"))
data = pd.read_csv("employee.csv")

st.set_page_config(page_title="Employee Performance Prediction")

st.title("Employee Performance Prediction System")

st.write("Search employee by ID or Name")

# Input
emp_input = st.text_input("Enter Employee ID or Name")

if st.button("Predict"):

    if emp_input == "":
        st.warning("Please enter Employee ID or Name")

    else:
        # Find employee
        if emp_input.isdigit():
            emp = data[data["Employee_ID"] == int(emp_input)]
        else:
            emp = data[data["Name"].str.lower() == emp_input.lower()]

        if emp.empty:
            st.error("Employee not found")

        else:
            # Features for prediction
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

            # Display results
            st.success(f"Performance Prediction: {prediction}")
            st.info(f"Performance Score: {score}%")
            st.warning(f"Risk Level: {risk}")
            st.success(f"HR Recommendation: {recommendation}")
            st.write(f"Experience Level: {exp_level}")
