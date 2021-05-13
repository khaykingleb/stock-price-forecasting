import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "DejaVu Sans",
    "font.sans-serif": ["Benton Sans"]
})

from IPython.display import clear_output

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 8

import numpy as np

import torch
from torch import nn
from torch import optim

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def plot_metric(train_metric, test_metric, title):
    """
    """
    plt.title(title, pad=10, fontsize=18, loc='left')
    
    plt.ylabel('Loss', labelpad=10, fontsize=14)
    plt.xlabel('Epoch', labelpad=10, fontsize=14)
    
    if title == 'RMSE':
        plt.plot(np.sqrt(train_metric), label='Train', zorder=1)
        plt.plot(np.sqrt(test_metric), c='orange', label='Test', zorder=2)
    else: 
        plt.plot(train_metric, label='Train', zorder=1)
        plt.plot(test_metric, c='orange', label='Test', zorder=2)
        
    plt.legend()
    
    plt.show()

 
def train_one_epoch(model, X, y_true, criterion, optimizer):
    """
    """
    model.train()
    
    y_pred = model(X.to(device))

    loss = criterion(y_pred.unsqueeze(1), y_true.unsqueeze(1).to(device))
    train_loss = loss.item()
    
    # Helps prevent the exploding gradient problem in RNNs / LSTMs
    nn.utils.clip_grad_norm_(model.parameters(), 5)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return train_loss

  
def predict(model, X, y_true, criterion):
    """
    """
    model.eval()
    y_pred = model(X.to(device))
    loss = criterion(y_pred.unsqueeze(1), y_true.unsqueeze(1).to(device))
    test_loss = loss.item()

    return test_loss


def train(model, X_train, X_test, y_train, y_test, criterion, optimizer, 
          n_epochs=10, scheduler=None):
    """
    """
    model.to(device)

    history_train_loss_by_epoch = []
    history_test_loss_by_epoch = []

    for epoch in range(n_epochs):
        clear_output(wait=True)
        print(f"Epoch: {epoch}") 
        history_train_loss_by_epoch.append(train_one_epoch(model, X_train, y_train,
                                                           criterion, optimizer))
        
        history_test_loss_by_epoch.append(predict(model, X_test, y_test, criterion))
      
        plot_metric(history_train_loss_by_epoch, history_test_loss_by_epoch, "RMSE")

        print(f"Test loss: {history_test_loss_by_epoch[-1]:.4}")
  
  
class LongShortTermMemoryWithout(nn.Module):
    """
    """
    def __init__(self, hidden_size=256, num_layers_LSTM=2, dropout_LSTM=0, dropout_FC=0.5):
        super(LongShortTermMemoryWithout, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers_LSTM = num_layers_LSTM

        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, 
                            num_layers=num_layers_LSTM, dropout=dropout_LSTM, 
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers_LSTM, x.size(0), self.hidden_size).to(device).requires_grad_()
        c_0 = torch.zeros(self.num_layers_LSTM, x.size(0), self.hidden_size).to(device).requires_grad_()

        out, (h_n, c_n) = self.lstm(x, (h_0.detach(), c_0.detach()))

        out = self.fc(out[:, -1, :])

        return out
