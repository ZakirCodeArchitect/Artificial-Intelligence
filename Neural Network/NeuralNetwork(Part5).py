import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

# Function to initialize weights
def init_weights_random(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)

# Initialize model with random weight initialization
model_random = Model(X_train.shape[1], 5, 2)
model_random.apply(init_weights_random)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()

# Different learning rates
learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001]

# Training Loop and Separate Plotting for each learning rate
for lr in learning_rates:
    optimizer_random = torch.optim.Adam(model_random.parameters(), lr=lr)
    losses_random = []

    for epoch in range(200):
        y_pred_random = model_random.forward(X_train)
        loss_random = criterion(y_pred_random, y_train)
        losses_random.append(loss_random.item())

        optimizer_random.zero_grad()
        loss_random.backward()
        optimizer_random.step()

        if epoch % 5 == 0:
            print(f'Epoch: {epoch}, Learning Rate: {lr}, Loss (Random): {loss_random.item()}')

    # Plotting Loss vs. Iterations for each learning rate
    plt.figure(figsize=(6, 4))
    plt.plot(range(200), losses_random, label=f'LR={lr}')
    plt.title(f'Loss vs. Iterations for LR={lr}')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
