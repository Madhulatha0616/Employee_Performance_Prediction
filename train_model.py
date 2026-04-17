import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("employee.csv")

# Features & Target
X = data[['Age','Experience','Job_Satisfaction','Work_Hours','Training_Hours']]
y = data['Performance']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Check accuracy (for reference only)
accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
print("Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open("employee_model.pkl", "wb"))

print("Model trained and saved successfully")
