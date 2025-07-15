import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
import numpy as np

if torch.backends.mps.is_available:
  device = "mps"
else:
  device = "cpu"

print(f"device is {device}")

class LinearRegressionModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_layer = nn.Linear(in_features = 1, out_features = 1)

  def forward(self, x: torch.tensor) -> torch.Tensor:
    return self.linear_layer(x)


weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

x = torch.arange(start, end, step).unsqueeze(dim = 1) # 将最后一个维度设为1，方便进行线性运算
y = weight * x + bias

train_split = int(0.8 * len(x))
test_split = int(0.2 * len(x))

x_train = x[:train_split]
y_train = y[:train_split]
x_test = x[-test_split:]
y_test = y[-test_split:]

def plot_predictions(train_data = x_train,
                     train_labels = y_train,
                     test_data = x_test,
                     test_labels = y_test,
                     predictions = None):
  plt.figure(figsize = (10, 7))
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  plt.scatter(test_data, test_labels, c="g", s=4, label="Test data")
  if predictions is not None:
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
  plt.legend(prop={"size": 14})

torch.manual_seed(42)
model_0 = LinearRegressionModel()

model_0 = model_0.to(device)
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

epochs = 1000
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.001)
epochs_count = []
train_loss = []
test_loss = []
lass_loss = -1

for epochs in range(epochs):
  model_0.train() # 训练
  y_pred = model_0(x_train)
  loss = loss_fn(y_pred, y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  model_0.eval()
  if (epochs % 10 == 0):
    with torch.inference_mode():
      y_test_pred = model_0(x_test)
      loss_test = loss_fn(y_test_pred, y_test)
      train_loss.append(loss)
      test_loss.append(loss_test)
      epochs_count.append(epochs)
      # print(f"epochs = {epochs} | train_loss = {loss} | test_loss = {loss_test}")

plt.plot(epochs_count, torch.tensor(train_loss).cpu().numpy(), label = "Train loss")
plt.plot(epochs_count, torch.tensor(test_loss).cpu().numpy(), label = "Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()