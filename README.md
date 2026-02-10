# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="1116" height="757" alt="image" src="https://github.com/user-attachments/assets/bf1e1aad-fecb-4ecb-a3cf-1e1f04db0f82" />


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

```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('DL1.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values

print(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Name:karthikeyan M
# Register Number:212223040088
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,1)
        self.fc3=nn.Linear(1,1)
        self.relu=nn.ReLU()
        self.history = {'loss': []}
  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)

# Name:karthikeyan M
# Register Number:212223040088
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      loss=criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()
#Append loss inside the loop
      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')
loss_df = pd.DataFrame(ai_brain.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```

### Dataset Information
<img width="283" height="282" alt="image" src="https://github.com/user-attachments/assets/7f595725-7a60-4f6c-b95e-1e8763ca71f4" />



### OUTPUT
<img width="441" height="235" alt="Screenshot 2026-02-10 083230" src="https://github.com/user-attachments/assets/9afc7b92-d4c5-4918-a109-d6f41e66ef50" />


### Training Loss Vs Iteration Plot
<img width="869" height="692" alt="Screenshot 2026-02-10 083036" src="https://github.com/user-attachments/assets/8aa59892-213f-417a-8093-af9bedc62112" />



### New Sample Data Prediction
<img width="957" height="180" alt="Screenshot 2026-02-10 083220" src="https://github.com/user-attachments/assets/7cf8a178-04a5-41f9-adda-3e3283aabdaa" />



## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
