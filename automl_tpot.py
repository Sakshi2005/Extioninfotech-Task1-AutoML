from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TPOT classifier
tpot = TPOTClassifier(verbosity=2, generations=5, population_size=20, random_state=42)

# Train TPOT classifier on the training data
tpot.fit(X_train, y_train)

# Evaluate the model on the test data
print(f"Test accuracy: {tpot.score(X_test, y_test)}")

# Export the best model found as a Python script
tpot.export('tpot_best_model.py')


