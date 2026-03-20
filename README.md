# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Title: Developing a Neural Network Regression Model

Objective: To design and implement a neural network architecture capable of accurately predicting a continuous target variable by learning non-linear relationships within a multi-dimensional dataset.

Scope: The model must utilize optimized backpropagation and loss functions (such as Mean Squared Error) to minimize prediction variance and ensure high generalization performance on unseen data.

Outcome: The final system will provide a robust, data-driven framework for precise numerical forecasting, outperforming traditional linear regression methods in handling complex, high-frequency data pattern

## Neural Network Model
Include the neural network model diagram.
<img width="953" height="586" alt="image" src="https://github.com/user-attachments/assets/a00cb1ff-9538-4ea1-9f69-238fd16c5a4a" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: karthikeyan M

### Register Number: 212223040088

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

```
```python
dataset1 = pd.read_csv('/content/dataset1 - Sheet1.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values
print(dataset1.head())

```
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

```
```python
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

```
```python
# Name: karthikeyan M
# Register Number: 212223040088
class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(1,8)
    self.fc2 = nn.Linear(8,10)
    self.fc3 = nn.Linear(10,1)
    self.relu = nn.ReLU()
    self.history = {'loss':[]}

  def forward(self,x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x

```
```python
# Initialize the Model, Loss Function, and Optimizer

ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr=0.001)

```
```python
# Name:karthikeyan M
# Register Number: 212223040088
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    # Write your code here
    for epoch in range(epochs):
      optimizer.zero_grad()
      loss = criterion(ai_brain(X_train), y_train)
      loss.backward()
      optimizer.step()

      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```
```python
train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)
with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

```
```python
import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

```
```python
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')

```

### Dataset Information
Include screenshot of the generated data

<img width="325" height="540" alt="image" src="https://github.com/user-attachments/assets/b8e9c857-958b-4bbc-94c1-01a4ce9e66af" />


### OUTPUT

### Training Loss Vs Iteration Plot
Include your plot here

<img width="799" height="754" alt="image" src="https://github.com/user-attachments/assets/b2c1d471-3767-46d4-98b8-726a2f1ae8ea" />


### New Sample Data Prediction
Include your sample input and output here

<img width="774" height="493" alt="image" src="https://github.com/user-attachments/assets/6015e758-9583-47f1-af05-60e8f3886b6f" />

<img width="898" height="131" alt="image" src="https://github.com/user-attachments/assets/80a479a6-74d9-4607-aeb5-1220379b6559" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
