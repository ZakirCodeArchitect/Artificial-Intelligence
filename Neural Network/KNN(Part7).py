import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a synthetic dataset for demonstration
X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try different values of k
k_values = np.arange(1, 5)  # Adjust the range as needed
train_accuracies = []
test_accuracies = []

for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    
    # Training accuracy
    train_pred = knn_classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    train_accuracies.append(train_accuracy)
    
    # Testing accuracy
    test_pred = knn_classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_accuracies.append(test_accuracy)

# Plot the graph
plt.figure(figsize=(8, 6))
plt.plot(k_values, train_accuracies, marker='o', label='Training Accuracy')
plt.plot(k_values, test_accuracies, marker='o', label='Testing Accuracy')
plt.title('Training and Testing Accuracies vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()

# Report training and testing accuracies for each k value
for k, train_acc, test_acc in zip(k_values, train_accuracies, test_accuracies):
    print(f'k = {k}: Training Accuracy = {train_acc:.4f}, Testing Accuracy = {test_acc:.4f}')
