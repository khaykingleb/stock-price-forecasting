import matplotlib.pyplot as plt
from IPython.display import clear_output

import torch
from torch import nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def plot_metric(title, train_metric=None, test_metric=None, val_metric=None):
    plt.title(title, pad=10, fontsize=18, loc='left', fontweight='bold')
    plt.ylabel('Loss', labelpad=10, fontsize=14) 
    plt.xlabel('Epoch', labelpad=10, fontsize=14)

    if train_metric:
        plt.plot(train_metric, label='Train', zorder=1)

    if test_metric:
        plt.plot(test_metric, c='orange', label='Test', zorder=2)
      
    if val_metric:
        plt.plot(val_metric, label='Validation', zorder=3)
        
    plt.grid(False)
    plt.legend()
    plt.show()
    
    
def train_one_epoch(model, X, y_true, criterion, optimizer, device):
    model.train()
    y_pred = model(X.to(device))
    loss = criterion(y_pred.unsqueeze(1), y_true.unsqueeze(1).to(device))
    train_loss = loss.item()
    
    # Prevent the exploding gradient problem
    nn.utils.clip_grad_norm_(model.parameters(), 5)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return train_loss
  
    
def predict(model, X, y_true, criterion, device):
    model.eval()
    y_pred = model(X.to(device))
    loss = criterion(y_pred.unsqueeze(1), y_true.unsqueeze(1).to(device))
    test_loss = loss.item()

    return test_loss


def train(model, criterion, optimizer, device, X_train, y_train, X_test=None, y_test=None, 
          n_epochs=10, verbose=True, return_loss_history=True, compute_test_loss=True):
    model.to(device)

    history_train_loss_by_epoch = []
    history_test_loss_by_epoch = []

    for epoch in range(n_epochs):
        history_train_loss_by_epoch.append(train_one_epoch(model, X_train, y_train,
                                                           criterion, optimizer, 
                                                           device))
        
        if compute_test_loss:
            history_test_loss_by_epoch.append(predict(model, X_test, y_test, 
                                                      criterion, device))
            
        if verbose:
            clear_output(wait=True)
            print(f"Epoch: {epoch + 1}") 

            plot_metric(f'{criterion.__class__.__name__}', 
                        history_train_loss_by_epoch, 
                        history_test_loss_by_epoch)
            
            print(f"Train loss: {history_train_loss_by_epoch[-1]:.4}")

            if compute_test_loss:
                print(f"Test loss: {history_test_loss_by_epoch[-1]:.4}")

    if return_loss_history:
        return history_train_loss_by_epoch, history_test_loss_by_epoch


class LSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, 
                 num_layers: int = 2, dropout: float = 0):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, dropout=dropout, 
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device).requires_grad_()
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device).requires_grad_()

        out, (h_n, c_n) = self.lstm(x, (h_0.detach(), c_0.detach()))

        out = self.fc(out[:, -1, :])

        return out
    
    
class GRU(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, 
                 num_layers: int = 2, dropout: float = 0):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                          num_layers=num_layers, dropout=dropout,
                          batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device).requires_grad_()
        out, (h_n) = self.gru(x, (h_0.detach()))

        out = self.fc(out[:, -1, :])

        return out
    
