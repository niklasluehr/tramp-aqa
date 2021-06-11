import torch
import torch.nn as nn
from opts import *

torch.manual_seed(random_seed); torch.cuda.manual_seed_all(random_seed)

class BiLSTM(nn.Module):
    def __init__(self, input_dim):
        super(BiLSTM, self).__init__()
        self.rnn = nn.LSTM(input_dim, 128, 1, batch_first=True, bidirectional=True)
        self.fc_final_score = nn.Linear(256,1)

    def forward(self, x):
        state = None
        lstm_output, state = self.rnn(x, state)
        final_score = self.fc_final_score(lstm_output[:,-1,:])
        return final_score

class BiLSTMDropout(nn.Module):
    def __init__(self, input_dim):
        super(BiLSTMDropout, self).__init__()
        self.rnn = nn.LSTM(input_dim, 256, 1, batch_first=True, bidirectional=True)
        self.fc_final_score = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        state = None
        lstm_output, state = self.rnn(x, state)
        final_score = self.fc_final_score(self.dropout(lstm_output[:,-1,:]))
        return final_score