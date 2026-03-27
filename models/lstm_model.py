import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

class NasdaqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def create_sequences(X, y, seq_length=5):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(Xs), np.array(ys)


def train_model(X, y):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    X_seq, y_seq = create_sequences(X_scaled, y_scaled)

    X_t = torch.FloatTensor(X_seq)
    y_t = torch.FloatTensor(y_seq)

    model = NasdaqLSTM(input_size=X_t.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(10):
        optimizer.zero_grad()
        output = model(X_t)
        loss = loss_fn(output, y_t)
        loss.backward()
        optimizer.step()

    return model, scaler_X, scaler_y