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

# Function to initialize weights with zeros
def init_weights_zeros(m):
    if type(m) == nn.Linear:
        nn.init.zeros_(m.weight)

# Function to initialize weights with small random values
def init_weights_random(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)

# Function to train the model
def train_model(model, optimizer, criterion):
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

    return losses

# Initialize models with different weight initialization methods
model_zeros = Model(X_train.shape[1], 5, 2)
model_random = Model(X_train.shape[1], 5, 2)

# Apply weight initialization
model_zeros.apply(init_weights_zeros)
model_random.apply(init_weights_random)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer_zeros = torch.optim.Adam(model_zeros.parameters(), lr=0.01)
optimizer_random = torch.optim.Adam(model_random.parameters(), lr=0.01)

# Train the models
losses_zeros = train_model(model_zeros, optimizer_zeros, criterion)
losses_random = train_model(model_random, optimizer_random, criterion)

# Plotting Loss vs. Iterations for both models
plt.figure(figsize=(10, 6))

plt.plot(range(200), losses_zeros, label='Zero Initialization')
plt.plot(range(200), losses_random, label='Random Initialization')

plt.title('Loss vs. Iterations for Different Weight Initializations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()
