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
    z_size: int
    batch_size: int

    def __init__(self, config: Config):
        for key, value in vars(config).items():
            if key in self.__annotations__:
                setattr(self, key, value)

        self.pred_model = VAE(
            self.s_size, self.z_size, 1, self.hidden_size
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
        o, h_out, pred_s = self.pred_present(s, a_lst, h_in, self.p_iters)
        # print(o.shape, h_out.shape)
        # print(pred_s.shape)
        pi = self.policy.pi(o)
        action = Categorical(pi).sample()
        return action.detach(), pi.detach(), h_out.detach(), pred_s.detach()

    def pred_present(self, s, a, h_in, iters):
        # 這裡是假設會以batch的形式輸入
        # shape => (seq_len, batch, data_size)
        # sample_action的輸入形狀為(1, 1, data_size)

        # 將輸入分成幾個mini_batch
        # 最後再將mini_batch組合起來
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
                    pred_s, phi_x_t, phi_z_t = self.pred_model(pred_s, a_lst[i], h_ti)
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
        # o_ti = torch.cat([o_ti, phi_z_t], dim=-1)
        return o_ti, h_first, s_ti
