import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from util import CrossAttention
from config import Config

class PredictiveModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, out_dim):
        super().__init__()
        self.in_dim = state_dim + action_dim + hidden_dim
        self.out_dim = out_dim

        self.attention = CrossAttention(
            state_dim,
            action_dim + hidden_dim,
            64,
            self.in_dim,
            1,
            0.0
        )
        self.D = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim)
            # CrossAttention(
            #     state_dim,
            #     action_dim + hidden_dim,
            #     64,
            #     out_dim,
            #     4,
            #     0.0
            # )
        )
        self.N = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim)
            # CrossAttention(
            #     state_dim,
            #     action_dim + hidden_dim,
            #     64,
            #     out_dim,
            #     4,
            #     0.0
            # )
        )
        self.F = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim),
            # CrossAttention(
            #     state_dim,
            #     action_dim + hidden_dim,
            #     64,
            #     out_dim,
            #     4,
            #     0.0
            # ),
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

        query, context = state, torch.cat([action, gru_out], dim=-1)
        x = (query, context)
        x = self.attention(x)
        # x = torch.cat([state, action, gru_out], dim=-1)
        delta = self.D(x) # D(s, a)
        adjusted_state = state + delta # s + D(s, a)

        new_state = self.N(x) # N(s, a)

        forget_weights = self.F(x) # F(s, a), in [0,1]

        pred_s = forget_weights * adjusted_state + (1 - forget_weights) * new_state
        # return pred_s.view(-1, 1, self.out_dim)
        return pred_s

class Policy(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.fc_pi = nn.Linear(input_dim, out_dim)
        self.fc_v  = nn.Linear(input_dim, 1)

    def pi(self, x):
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=-1)
        return prob

    def v(self, x):
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
    batch_size: int

    def __init__(self, config: Config):
        for key, value in vars(config).items():
            if key in self.__annotations__:
                setattr(self, key, value)

        self.rnn = RNN(self.s_size, 64, self.hidden_size)
        self.pred_model = PredictiveModel(self.s_size, 1, self.hidden_size, self.s_size)
        self.policy = Policy(self.hidden_size, self.a_size)

    def set_device(self, device: str):
        self.rnn.to(device)
        self.pred_model.to(device)
        self.policy.to(device)

    def load_params(self, state_dict: dict):
        self.rnn.load_state_dict(state_dict["rnn"])
        self.pred_model.load_state_dict(state_dict["pred_model"])
        self.policy.load_state_dict(state_dict["policy"])

    def output_params(self):
        return {
            "rnn": self.rnn.state_dict(),
            "pred_model": self.pred_model.state_dict(),
            "policy": self.policy.state_dict()
        }

    def sample_action(self, s, a_lst, h_in):
        o, h_out, pred_s = self.pred_present(s, a_lst, h_in, self.p_iters)
        # print(o.shape, h_out.shape)
        # print(pred_s.shape)
        pi = self.policy.pi(o)
        v = self.policy.v(o)
        action = Categorical(pi).sample()
        return action.detach(), pi.detach(), h_out.detach(), pred_s.detach(), v.detach()

    def pred_present(self, s, a, h_in, iters):
        # generate h_in for all s
        # (seq_len, batch=1, data_dim)
        o, _ = self.rnn(s, h_in)
        # print(o.shape)
        o = torch.cat([h_in, o[:-1]])
        # print(o.shape)

        # generate s_ti for all s
        # (seq_len=1, batch, data_dim)
        o_ti, s_ti = [], []
        s, h = s.transpose(0, 1), o.transpose(0, 1)
        # print(s.shape, a.shape, h.shape)
        s, h, a = torch.split(s, self.batch_size, dim=1), \
                    torch.split(h, self.batch_size, dim=1), \
                    torch.split(a, self.batch_size, dim=1)
        # print(len(s), len(h), len(a))

        for pred_s, h_in, a_lst in zip(s, h, a):
            mini_o_ti, mini_s_ti = [], []
            # print(a_lst.shape)
            a_lst = torch.split(a_lst, 1, dim=-1)
            # print(a_lst[0].shape)
            pred_o, h_ti = self.rnn(pred_s, h_in)
            # print(pred_o.shape, h_ti.shape)
            h_first = h_ti
            for i in range(iters):
                pred_s = self.pred_model(pred_s, a_lst[i], pred_o)
                # print(pred_s.shape)
                mini_s_ti.append(pred_s)
                pred_o, h_ti = self.rnn(pred_s, h_ti)
                mini_o_ti.append(pred_o)
                # print(pred_o.shape, h_ti.shape)
            mini_o_ti = torch.cat(mini_o_ti) if len(mini_o_ti) > 0 else torch.tensor([])
            mini_s_ti = torch.cat(mini_s_ti) if len(mini_s_ti) > 0 else torch.tensor([])
            # print(mini_o_ti.shape, mini_s_ti.shape)
            o_ti.append(mini_o_ti)
            s_ti.append(mini_s_ti)
        o_ti = torch.cat(o_ti, dim=1)[-1].unsqueeze(0) if iters > 0 else pred_o
        s_ti = torch.cat(s_ti, dim=1)
        return o_ti, h_first, s_ti

    # def pred_prob(self, s, a_lst, h_in):
    #     o, h_out, _ = self.pred_present(s, a_lst, h_in, self.p_iters)
    #     pi = self.policy.pi(o)
    #     return pi, h_out[:, 0].unsqueeze(0)

    # def pred_critic(self, s, h_in):
    #     o, _ = self.rnn(s, h_in)
    #     v = self.policy.v(o)
    #     return v

    def pred_prob_and_critic(self, s, h_in):
        o, _ = self.rnn(s, h_in)
        second_hidden = o[0].unsqueeze(0)
        pi = self.policy.pi(o)
        v = self.policy.v(o)
        return pi, v, second_hidden.detach()

    # def pred_prob_and_critic_batch(self, s, s_offset, h_in):
    #     # generate all starting h_in
    #     # (seq_len, batch, data_dim)
    #     o, _ = self.rnn(s, h_in)
    #     second_hidden = o[0].unsqueeze(0)
    #     o = torch.cat([h_in, o[:-1]])

    #     # generate o for all s_offset
    #     # (seq_len, batch, data_dim)
    #     all_h_in = o.transpose(0, 1)

    #     o, _ = self.rnn(s_offset, all_h_in)
    #     h_first = o[0]
    #     last_o = o[-1]
    #     pi = self.policy.pi(last_o)
    #     v = self.policy.v(last_o)
    #     return pi, v, h_first, second_hidden

    # def pred_prob_and_critic_old(self, s, h_in):
    #     # predicting v(s_{t+d}) at t
    #     # s is true state here, instead of those predicted by P
    #     h_first = None
    #     for i in range(len(s)):
    #         o_ti, h_out = self.rnn(s[i], h_in)
    #         h_in = h_out
    #         if h_first is None:
    #             h_first = h_out
    #     pi = self.policy.pi(o_ti)
    #     v = self.policy.v(o_ti)
    #     return pi, v, h_first