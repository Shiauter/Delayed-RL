import torch
import torch.nn as nn
import torch.nn.functional as F

from util import Categorical
from config import Config

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
        # s = (batch, s_size)
        # a = (batch, 1)
        # o = (batch, hidden_size)
        # state = state.view(-1)
        # action = action.view(-1)
        # gru_out = gru_out.view(-1)

        x = torch.cat([state, action, gru_out], dim=-1)
        delta = self.D(x) # D(s, a)
        adjusted_state = state + delta # s + D(s, a)

        new_state = self.N(x) # N(s, a)

        forget_weights = self.F(x) # F(s, a), in [0,1]

        pred_s = forget_weights * adjusted_state + (1 - forget_weights) * new_state
        # return pred_s.view(-1, 1, self.out_dim)
        return pred_s

class Policy(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1   = nn.Linear(input_dim, hidden_dim)
        self.fc_pi = nn.Linear(hidden_dim, out_dim)
        self.fc_v  = nn.Linear(hidden_dim, 1)

    def pi(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=-1)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1   = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, out_dim)

    def forward(self, x, h):
        x = F.relu(self.fc1(x))
        # x = x.view(-1, 1, self.hidden_dim)
        out, hidden = self.rnn(x, h)
        return out, hidden

class Actor:
    s_size: int
    a_size: int
    delay: int
    p_iters: int
    T_horizon: int
    hidden_size: int

    def __init__(self, config: Config):
        for key, value in vars(config).items():
            if key in self.__annotations__:
                setattr(self, key, value)

        self.rnn = RNN(self.s_size, 128, 64)
        self.pred_model = PredictiveModel(self.s_size, 1, 64, self.s_size)
        self.policy = Policy(64, 32, self.a_size)

        self.dist = Categorical((self.a_size,))

    def load_params(self, state_dict: list[dict]):
        self.rnn.load_state_dict(state_dict[0])
        self.pred_model.load_state_dict(state_dict[1])
        self.policy.load_state_dict(state_dict[2])

    def output_params(self):
        return [
            self.rnn.state_dict(),
            self.pred_model.state_dict(),
            self.policy.state_dict()
        ]

    def sample_action(self, s, a_lst, h_in):
        o, h_out, pred_s = self.pred_present(s, a_lst, h_in, self.p_iters)
        # o and h has same shape is weird
        pi = self.policy.pi(o)
        self.dist.set_probs(pi)
        action = self.dist.sample()
        return action.item(), pi, h_out, pred_s

    def pred_present(self, s, a_lst, h_in, iters):
        # generate all starting h_in
        # (seq_len, batch=1, data_dim)
        o, _ = self.rnn(s, h_in)
        o = torch.cat([h_in, o[:-1]])

        # generate s_ti for all s
        # (seq_len=1, batch, data_dim)
        s_ti = []
        pred_s = s.transpose(0, 1)
        all_h_in = o.transpose(0, 1)

        o_ti, h_ti = self.rnn(pred_s, all_h_in)
        h_first = h_ti
        for i in range(iters):
            pred_s = self.pred_model(pred_s, a_lst[:, :, i].unsqueeze(-1), o_ti)
            s_ti.append(pred_s)
            o_ti, h_ti = self.rnn(pred_s, h_ti)
        s_ti = torch.cat(s_ti) if len(s_ti) > 0 else torch.tensor([])
        return o_ti, h_first, s_ti

    def pred_prob(self, s, a_lst, h_in):
        o, h_out, _ = self.pred_present(s, a_lst, h_in, self.p_iters)
        pi = self.policy.pi(o)
        return pi, h_out[:, 0].unsqueeze(0)

    def pred_critic(self, s, h_in):
        o, _ = self.rnn(s, h_in)
        v = self.policy.v(o)
        return v

    def pred_prob_and_critic(self, s, h_in):
        o, _ = self.rnn(s, h_in)
        second_hidden = o[0].unsqueeze(0)
        o = o[self.delay:]
        pi = self.policy.pi(o)
        v = self.policy.v(o)
        return pi, v, second_hidden

    def pred_prob_and_critic_batch(self, s, s_offset, h_in):
        # generate all starting h_in
        # (seq_len, batch, data_dim)
        o, _ = self.rnn(s, h_in)
        second_hidden = o[0].unsqueeze(0)
        o = torch.cat([h_in, o[:-1]])

        # generate o for all s_offset
        # (seq_len, batch, data_dim)
        all_h_in = o.transpose(0, 1)

        o, _ = self.rnn(s_offset, all_h_in)
        h_first = o[0]
        last_o = o[-1]
        pi = self.policy.pi(last_o)
        v = self.policy.v(last_o)
        return pi, v, h_first, second_hidden

    def pred_prob_and_critic_old(self, s, h_in):
        # predicting v(s_{t+d}) at t
        # s is true state here, instead of those predicted by P
        h_first = None
        for i in range(len(s)):
            o_ti, h_out = self.rnn(s[i], h_in)
            h_in = h_out
            if h_first is None:
                h_first = h_out
        pi = self.policy.pi(o_ti)
        v = self.policy.v(o_ti)
        return pi, v, h_first