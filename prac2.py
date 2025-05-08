import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Generate synthetic data
def generate_structural_data(n=200):
    X = np.random.rand(n, 2)
    y = ((X[:, 0] + X[:, 1]) > 1).astype(int)
    return X, y

# Clonal Selection Classifier (simplified)
def clonal_selection_classify(X_train, y_train, n_detectors=10):
    np.random.seed(42)
    
    # Initialize detectors (antibodies) as random samples from each class
    detectors = []

    for cls in [0, 1]:  # Loop over both classes
        class_samples = X_train[y_train == cls]  # Get all samples from the current class
        for _ in range(n_detectors):  # Create 'n_detectors' for each class
            random_sample = class_samples[np.random.randint(len(class_samples))]  # Pick a random sample
            detectors.append((random_sample, cls))  # Add the detector (sample, class) pair
    
    return detectors # Added return statement to return the detectors

# Predict with the classifiers (using Scikit-learn's KNN for prediction)
def predict(X_test, detectors):
    # KNN classifier to predict class based on distance
    knn = KNeighborsClassifier(n_neighbors=3)
    
    # Training the KNN with synthetic data
    X_train, y_train = np.array([det_vec for det_vec, _ in detectors]), np.array([det_label for _, det_label in detectors])
    knn.fit(X_train, y_train)
    """
    X_train = np.array([[0.1, 0.2], [0.3, 0.4]])  # 2D array of feature vectors
    y_train = np.array([0, 1])  # 1D array of class labels
"""
    # Predict using the trained KNN model
    return knn.predict(X_test)

# Main process
X, y = generate_structural_data(300)  # Generate synthetic data

# Use Scikit-learn's train_test_split to split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier (substitute Clonal Selection with detector initialization)
detectors = clonal_selection_classify(X_train, y_train)

# Predict using the trained model
y_pred = predict(X_test, detectors)

# Accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")

# Plot the decision boundaries
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolor='k')
plt.title("Structural Damage Classification (Simplified CSA)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()








































































