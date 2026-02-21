from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Scikit-learn uses 'multinomial' by default for multi-class problems
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X, y)

# Predict the species for a new flower
prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
print(f"Predicted Species Index: {prediction[0]}")