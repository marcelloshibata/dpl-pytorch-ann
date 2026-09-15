#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled)
print(y_train)

#%%
X_train_scaled_tensor = torch.from_numpy(X_train_scaled).float()
X_test_scaled_tensor = torch.from_numpy(X_test_scaled).float()
y_train_tensor = torch.from_numpy(y_train).unsqueeze(1).float() # aumentar o numero de dimensoes para se encaixar
y_test_tensor = torch.from_numpy(y_test).unsqueeze(1).float()

print(X_train_scaled_tensor.shape) # Saber a quantidade de dados para o batch-size (pulos)

train_dataset = TensorDataset(X_train_scaled_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#%%
class BCNet(nn.Module):
    def __init__(self):
        super(BCNet, self).__init__()

        self.fc1 = nn.Linear(30, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))

        return x

model = BCNet()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
epochs = 20

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()

        preds = model(x_batch)
        loss = criterion(preds, y_batch)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch {epoch+1}: Loss was {running_loss / len(train_loader)}')
#%%
with torch.no_grad():
    model.eval()
    preds = model(X_test_scaled_tensor)
    loss = criterion(preds, y_test_tensor).item()

    accuracy = ((preds >= 0.5) == y_test_tensor).float().mean().item()

print(accuracy)