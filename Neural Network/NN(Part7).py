import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load the dataset
url = '/content/drive/Othercomputers/HP Intel core/Semester 5/G-Drive Courses/CS-414 AI V/Assignment 2/clean_dataset.csv'
df = pd.read_csv(url)

# Data Preprocessing
categorical_cols = ['Industry', 'Ethnicity', 'Citizen']
numerical_cols = [col for col in df.columns if col not in ['Approved'] + categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

X = preprocessor.fit_transform(df.drop('Approved', axis=1))
y = df['Approved'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Define the Model Class
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

# Function to train the model with different hidden layer sizes
def train_model(hidden_size):
    model = Model(X_train.shape[1], hidden_size, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(200):
        optimizer.zero_grad()
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

    return model, losses

# Function to calculate accuracy
def calculate_accuracy(model, X_data, y_data):
    with torch.no_grad():
        predictions = torch.argmax(model(X_data), dim=1)
        accuracy = (predictions == y_data).float().mean().item()
    return accuracy

# Train the model for different hidden layer sizes and calculate accuracies
hidden_sizes = [2, 3, 4, 5]
train_accuracies = []
test_accuracies = []
losses_dict = {}

plt.figure(figsize=(10, 6))

for size in hidden_sizes:
    trained_model, losses = train_model(size)
    losses_dict[size] = losses
    
    train_acc = calculate_accuracy(trained_model, X_train, y_train)
    test_acc = calculate_accuracy(trained_model, X_test, y_test)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    
    # Plotting Loss vs. Iterations for different Hidden Layer Sizes
    plt.plot(range(200), losses, label=f'Hidden Size: {size}')

plt.title('Loss vs. Iterations for different Hidden Layer Sizes')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting Training and Testing Accuracies for different Hidden Layer Sizes
plt.figure(figsize=(8, 6))
plt.plot(hidden_sizes, train_accuracies, marker='o', linestyle='-', label='Training Accuracy')
plt.plot(hidden_sizes, test_accuracies, marker='o', linestyle='-', label='Testing Accuracy')
plt.title('Training and Testing Accuracies for different Hidden Layer Sizes')
plt.xlabel('Hidden Layer Size')
plt.ylabel('Accuracy')
plt.xticks(hidden_sizes)
plt.legend()
plt.grid(True)
plt.show()

# Report training and testing accuracies for each hidden layer size
for size, train_acc, test_acc in zip(hidden_sizes, train_accuracies, test_accuracies):
    print(f'Hidden Layer Size = {size}: Training Accuracy = {train_acc:.4f}, Testing Accuracy = {test_acc:.4f}')
