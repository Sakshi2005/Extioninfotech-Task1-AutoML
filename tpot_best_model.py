import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tpot import TPOTClassifier
import pickle



# Importing necessary libraries
from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TPOT
tpot = TPOTClassifier(verbosity=2, generations=5, population_size=20, random_state=42)

# Fit TPOT
try:
    tpot.fit(X_train, y_train)
    print("Model fitted successfully.")
except Exception as e:
    print(f"Error fitting the model: {e}")

# Check if the model is fitted
if hasattr(tpot, 'fitted_pipeline_'):
    print("TPOT model is fitted.")
    
    # Make predictions
    y_pred = tpot.predict(X_test)
    print("Predictions:", y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
else:
    print("Cannot make predictions; model is not fitted.")

# Exporting the best pipeline
tpot.export('tpot_best_pipeline.py')

# Save the TPOT model
joblib.dump(tpot.fitted_pipeline_, 'tpot_model.pkl')
print("Model saved as tpot_model.pkl.")

# Load the model
model = joblib.load('tpot_model.pkl')

# Example of making predictions
new_data = [[5.1, 3.5, 1.4, 0.2]]  # Replace with actual new data
predictions = model.predict(new_data)
print("Predictions for new data:", predictions)