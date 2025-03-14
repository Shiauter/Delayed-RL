import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from vae import VAE
from util import CrossAttention
from config import Config

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

        z_dim = self.hidden_size // 2
        self.pred_model = VAE(
            self.s_size, z_dim, 1, self.hidden_size
        )
        self.rnn = RNN(self.s_size + z_dim, 64, self.hidden_size)
        self.policy = Policy(self.hidden_size, self.a_size)

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
        # print(o.shape, h_out.shape)
        # print(pred_s.shape)
        pi = self.policy.pi(o)
        action = Categorical(pi).sample()
        return action.detach(), pi.detach(), h_out.detach(), pred_s.detach()

    def pred_present(self, s, a, h_in, iters):
        if iters > 0:
            o_ti, s_ti = [], []
            s, h, a = torch.split(s, self.batch_size, dim=1), \
                        torch.split(h_in, self.batch_size, dim=1), \
                        torch.split(a, self.batch_size, dim=1)

            for pred_s, h_in, a_lst in zip(s, h, a):
                mini_o_ti, mini_s_ti = [], []
                a_lst = torch.split(a_lst, 1, dim=-1)
                h_first, h_ti = None, h_in
                for i in range(iters):
                    pred_s, phi_x_t, z_t, phi_z_t = self.pred_model(pred_s, a_lst[i], h_ti)
                    pred_o, h_ti = self.rnn(torch.cat([phi_x_t, phi_z_t], dim=-1), h_ti)
                    mini_s_ti.append(pred_s)
                    mini_o_ti.append(pred_o)
                    if h_first is None:
                        h_first = h_ti
                mini_o_ti = torch.cat(mini_o_ti) if len(mini_o_ti) > 0 else torch.tensor([])
                mini_s_ti = torch.cat(mini_s_ti) if len(mini_s_ti) > 0 else torch.tensor([])
                o_ti.append(mini_o_ti)
                s_ti.append(mini_s_ti)
            o_ti = torch.cat(o_ti, dim=1)[-1].unsqueeze(0) if iters > 0 else pred_o
            s_ti = torch.cat(s_ti, dim=1)
        return o_ti, h_first, s_ti

    def pred_prob_and_critic(self, s, h_in):
        o, _ = self.rnn(s, h_in)
        second_hidden = o[0].unsqueeze(0)
        pi = self.policy.pi(o)
        v = self.policy.v(o)
        return pi, v, second_hidden.detach()
