from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Scikit-learn uses 'multinomial' by default for multi-class problems
model = LogisticRegression(solver='lbfgs', max_iter=200)
model.fit(X, y)

# Predict the species for a new flower
prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
prediction2 = model.predict([[5.9, 3.0, 5.1, 1.8]])
prediction3 = model.predict([ [5.0, 2.3, 3.3, 1.0 ]])

print(f"Predicted Species Index: {prediction[0]}")
print(f"Predicted Species Index: {prediction2[0]}")
print(f"Predicted Species Index: {prediction3[0]}")