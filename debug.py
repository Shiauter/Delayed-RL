import torch.nn as nn
import torch
import torch.nn.functional as F

class PredictiveModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, out_dim):
        super().__init__()
        self.in_dim = state_dim + action_dim + hidden_dim
        self.out_dim = out_dim

        self.D = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim)
        )
        self.N = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim)
        )
        self.F = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim),
            nn.Sigmoid()
        )
        self.criterion = nn.MSELoss()

    def forward(self, state, action, gru_out):
        # s = (batch, state_dim)
        # a = (batch, action_dim)
        # o = (batch, hidden_dim)

        x = torch.cat([state, action, gru_out], dim=-1)
        print("cat:",x.shape)
        delta = self.D(x) # D(s, a)
        adjusted_state = state + delta # s + D(s, a)

        new_state = self.N(x) # N(s, a)

        forget_weights = self.F(x) # F(s, a), in [0,1]

        pred_s = forget_weights * adjusted_state + (1 - forget_weights) * new_state
        return pred_s # (batch, out_dim)

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1   = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, out_dim)

    def forward(self, x, h):
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        # x = x.view(-1, 1, self.hidden_dim)
        out, hidden = self.rnn(x, h)
        return out, hidden


pd = PredictiveModel(4, 1, 2, 4)
rnn = RNN(4, 8, 2)

s = torch.randn(5,1,4)
a =  torch.randn(1,5,10)
h =  torch.randn(1,1,2)

# s = torch.randn(4).unsqueeze(0)
# a =  torch.randn(1).unsqueeze(0)
# h =  torch.randn(1,1,2)

# print(a[:,:,1].unsqueeze(-1).shape)
# print(s.shape,a.shape,h.shape)

s_ti = []
o,h_t = rnn(s,h)
print("s:", s.shape)
print("a:",a[:,:,0].unsqueeze(-1).shape)
print("o:",o.shape)
print("h:",h_t.shape)
o = torch.cat([h,o[:-1]])
s = s.view(1, -1, 4)
o = o.view(1, -1, 2)
print("s:", s.shape)
print("a:",a[:,:,0].unsqueeze(-1).shape)
print("o:",o.shape)
x = pd(s,a[:,:,0].unsqueeze(-1),o)
print("x:",x.shape)
s_ti.append(x)
s = x
h = o
o,h_t = rnn(s,h)
print("o:",o.shape)
print("h:",h_t.shape)
x=pd(s,a[:,:,1].unsqueeze(-1),o)
print("x:",x.shape)
s_ti.append(x)
s_ti = torch.cat(s_ti)
print(s_ti.shape)
