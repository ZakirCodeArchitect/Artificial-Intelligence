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

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=41)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=41)

X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_val = torch.LongTensor(y_val)
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
learning_rate = 0.01
optimizer_random = torch.optim.Adam(model_random.parameters(), lr=learning_rate)

# Training Loop with Early Stopping
epochs = 200
losses_train = []
losses_val = []
best_val_loss = float('inf')
patience = 20  # Number of epochs with no improvement after which training will be stopped

for epoch in range(epochs):
    # Model with random weight initialization
    y_pred_train = model_random.forward(X_train)
    loss_train = criterion(y_pred_train, y_train)
    losses_train.append(loss_train.item())

    optimizer_random.zero_grad()
    loss_train.backward()
    optimizer_random.step()

    # Validation loss
    with torch.no_grad():
        y_pred_val = model_random.forward(X_val)
        loss_val = criterion(y_pred_val, y_val)
        losses_val.append(loss_val.item())

    # Early stopping check
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f'Early stopping after {epoch} epochs')
        break

    if epoch % 5 == 0:
        print(f'Epoch: {epoch}, Loss (Train): {loss_train.item()}, Loss (Validation): {loss_val.item()}')

# Plotting Loss vs. Iterations for Training and Validation
plt.figure(figsize=(8, 6))
plt.plot(range(len(losses_train)), losses_train, label='Training Loss', color='blue')
plt.plot(range(len(losses_val)), losses_val, label='Validation Loss', color='orange')
plt.title('Loss vs. Iterations with Early Stopping')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate on the test set
with torch.no_grad():
    y_pred_test = model_random.forward(X_test)
    test_loss = criterion(y_pred_test, y_test)
    print(f'Test Loss: {test_loss.item()}')
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

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=41)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=41)

X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_val = torch.LongTensor(y_val)
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
learning_rate = 0.01
optimizer_random = torch.optim.Adam(model_random.parameters(), lr=learning_rate)

# Training Loop with Early Stopping
epochs = 200
losses_train = []
losses_val = []
best_val_loss = float('inf')
patience = 20  # Number of epochs with no improvement after which training will be stopped

for epoch in range(epochs):
    # Model with random weight initialization
    y_pred_train = model_random.forward(X_train)
    loss_train = criterion(y_pred_train, y_train)
    losses_train.append(loss_train.item())

    optimizer_random.zero_grad()
    loss_train.backward()
    optimizer_random.step()

    # Validation loss
    with torch.no_grad():
        y_pred_val = model_random.forward(X_val)
        loss_val = criterion(y_pred_val, y_val)
        losses_val.append(loss_val.item())

    # Early stopping check
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f'Early stopping after {epoch} epochs')
        break

    if epoch % 5 == 0:
        print(f'Epoch: {epoch}, Loss (Train): {loss_train.item()}, Loss (Validation): {loss_val.item()}')

# Plotting Loss vs. Iterations for Training and Validation
plt.figure(figsize=(8, 6))
plt.plot(range(len(losses_train)), losses_train, label='Training Loss', color='blue')
plt.plot(range(len(losses_val)), losses_val, label='Validation Loss', color='orange')
plt.title('Loss vs. Iterations with Early Stopping')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate on the test set
with torch.no_grad():
    y_pred_test = model_random.forward(X_test)
    test_loss = criterion(y_pred_test, y_test)
    print(f'Test Loss: {test_loss.item()}')
