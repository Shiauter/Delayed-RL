import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from vae_for_vrnn import VAE
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
        out, hidden = self.rnn(x, h)
        return out, hidden

class Actor:
    s_size: int
    a_size: int
    delay: int
    p_iters: int
    T_horizon: int
    hidden_size: int
    z_size: int
    batch_size: int

    def __init__(self, config: Config):
        for key, value in vars(config).items():
            if key in self.__annotations__:
                setattr(self, key, value)

        self.pred_model = VAE(
            self.s_size, self.z_size, 1, self.hidden_size,
            config.pred_s_source, config.nll_include_const, config.set_std_to_1
        )
        self.rnn = RNN(self.s_size + self.z_size, 64, self.hidden_size)
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
        o, h_out = self.pred_present(s, a_lst, h_in)
        pi = self.policy.pi(o)
        v = self.policy.v(o)
        action = Categorical(pi).sample()
        return action, pi, h_out, v

    def pred_present(self, s, a_lst, h_in):
        # shape => (seq_len, batch, data_size)
        # pred_o, pred_h = self.rnn(s, h_in)
        # return pred_o, pred_h

        # inference
        phi_x, phi_z = self.pred_model(s, h_in)
        cond_in = torch.cat([phi_x, phi_z], dim=-1)
        pred_o, pred_h = self.rnn(cond_in, h_in)
        h_out = pred_h

        # rollout
        a_lst = torch.split(a_lst, 1, dim=-1)
        for p in range(self.p_iters):
            phi_x, phi_z = self.pred_model.rollout(a_lst[p], pred_h)
            cond_in = torch.cat([phi_x, phi_z], dim=-1)
            pred_o, pred_h = self.rnn(cond_in, pred_h)

        return pred_o, h_out