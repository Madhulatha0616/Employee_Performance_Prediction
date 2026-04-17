import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Employee Performance Prediction")

st.title("Employee Performance Prediction System")

# ------------------ SAFE FILE LOADING ------------------

def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), "employee_model.pkl")
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error("❌ Model file not found or failed to load")
        st.stop()

def load_data():
    try:
        data_path = os.path.join(os.path.dirname(__file__), "employee.csv")
        return pd.read_csv(data_path)
    except Exception as e:
        st.error("❌ CSV file not found or failed to load")
        st.stop()

model = load_model()
data = load_data()

# ------------------ INPUT ------------------

st.write("Search employee by ID or Name")

emp_input = st.text_input("Enter Employee ID or Name")

# ------------------ PREDICTION ------------------

if st.button("Predict"):

    if emp_input.strip() == "":
        st.warning("Please enter Employee ID or Name")

    else:
        try:
            # Find employee
            if emp_input.isdigit():
                emp = data[data["Employee_ID"] == int(emp_input)]
            else:
                emp = data[data["Name"].str.lower() == emp_input.lower()]

            if emp.empty:
                st.error("Employee not found")

            else:
                # Features
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

                # Display
                st.success(f"Performance Prediction: {prediction}")
                st.info(f"Performance Score: {score}%")
                st.warning(f"Risk Level: {risk}")
                st.success(f"HR Recommendation: {recommendation}")
                st.write(f"Experience Level: {exp_level}")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
