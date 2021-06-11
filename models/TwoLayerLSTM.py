import torch
import torch.nn as nn
from opts import *

torch.manual_seed(random_seed); torch.cuda.manual_seed_all(random_seed)

class TwoLayerLSTM(nn.Module):
    def __init__(self, input_dim):
        super(TwoLayerLSTM, self).__init__()
        self.rnn = nn.LSTM(input_dim, 128, 2, batch_first=True)
        self.fc_final_score = nn.Linear(128,1)

    def forward(self, x):
        state = None
        lstm_output, state = self.rnn(x, state)
        final_score = self.fc_final_score(lstm_output[:,-1,:])
        return final_score

class TwoLayerLSTMDropout(nn.Module):
    def __init__(self, input_dim):
        super(TwoLayerLSTMDropout, self).__init__()
        self.rnn = nn.LSTM(input_dim, 256, 2, dropout=.5, batch_first=True)
        self.fc_final_score = nn.Linear(256, 1)

    def forward(self, x):
        state = None
        lstm_output, state = self.rnn(x, state)
        final_score = self.fc_final_score(lstm_output[:,-1,:])
        return final_score