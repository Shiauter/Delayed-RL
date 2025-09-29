import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config_control_group import Config
from util import CrossAttention

class PredictiveModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, out_dim):
        super().__init__()
        self.action_dim = action_dim
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

    def forward(self, state, action, gru_out):
        action = F.one_hot(action.long(), self.action_dim).reshape(*action.shape[:-1], self.action_dim).float()
        x = torch.cat([state, action, gru_out], dim=-1)
        delta = self.D(x)
        adjusted_state = state + delta
        new_state = self.N(x)
        forget_weights = self.F(x)
        pred_s = forget_weights * adjusted_state + (1 - forget_weights) * new_state
        return pred_s

class Policy(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        h_dim = max(64, input_dim * 2)
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc_pi = nn.Linear(h_dim, out_dim)
        self.fc_v  = nn.Linear(h_dim, 1)

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

    used_method: str

    def __init__(self, config: Config):
        for key, value in vars(config).items():
            if key in self.__annotations__:
                setattr(self, key, value)

        self._init_networks()

    def set_device(self, device: str):
        if hasattr(self, "rnn"): self.rnn.to(device)
        if hasattr(self, "pred_model"): self.pred_model.to(device)
        self.policy.to(device)

    def load_params(self, state_dict: dict):
        if hasattr(self, "rnn"): self.rnn.load_state_dict(state_dict["rnn"])
        if hasattr(self, "pred_model"): self.pred_model.load_state_dict(state_dict["pred_model"])
        self.policy.load_state_dict(state_dict["policy"])

    def output_params(self):
        out = {}
        if hasattr(self, "rnn"): out["rnn"] = self.rnn.state_dict()
        if hasattr(self, "pred_model"): out["pred_model"] = self.pred_model.state_dict()
        out["policy"] = self.policy.state_dict()
        return out

    def sample_action(self, s, a_lst, h_in):
        if self.used_method in {"LSTM_PPO", "P_Model"}:
            o, h_out, _ = self.pred_present(s, a_lst, h_in)
            x = o# torch.cat([o, s], dim=-1)
        elif self.used_method == "PPO":
            x = s
            h_out = h_in # unused in algo.

        pi = self.policy.pi(x)
        v = self.policy.v(x)
        action = Categorical(pi).sample()
        return action, pi, h_out, v


    def pred_present(self, s, a_lst, h_in):
        # shape => (seq_len, batch, data_size)
        # pred_o, pred_h = self.rnn(s, h_in)
        # return pred_o, pred_h

        pred_o, pred_h = self.rnn(s, h_in)
        h_out = pred_h
        s_ti = s
        pred_s = []

        # rollout
        if self.used_method == "P_Model":
            a_lst = torch.split(a_lst, 1, dim=-1)
            for p in range(self.p_iters):
                s_ti = self.pred_model(s_ti, a_lst[p], pred_o)
                pred_o, pred_h = self.rnn(s_ti, pred_h)
                pred_s.append(s_ti)
        if len(pred_s) > 0: pred_s = torch.cat(pred_s, dim=1)
        return pred_o, h_out, pred_s

    def _init_networks(self):
        if self.used_method == "PPO":
            self.policy = Policy(self.s_size, self.a_size)
        elif self.used_method == "LSTM_PPO":
            self.rnn = RNN(self.s_size, 64, self.hidden_size)
            self.policy = Policy(self.hidden_size, self.a_size)
        elif self.used_method == "P_Model":
            self.rnn = RNN(self.s_size, 64, self.hidden_size)
            self.policy = Policy(self.hidden_size, self.a_size)
            self.pred_model = PredictiveModel(self.s_size, self.a_size, self.hidden_size, self.s_size)
        else:
            raise KeyError(f"Unknown used_method: {self.used_method}")