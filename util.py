import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys, shutil, math

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, h_dim, out_dim, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.dim_per_head = h_dim // n_heads
        assert h_dim % n_heads == 0, "Invalid numbers of heads"

        self.q_net = nn.Linear(query_dim, h_dim)
        self.k_net = nn.Linear(context_dim, h_dim)
        self.v_net = nn.Linear(context_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, out_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

    def forward(self, x):
        query, context = x
        B, T_q, C_q = query.shape  # (batch size, query length, query dim)
        _, T_c, C_c = context.shape  # (batch size, context length, context dim)
        H = self.n_heads  # Number of heads
        D = self.dim_per_head  # Dimension per head

        # Project Q, K, V
        q = self.q_net(query).view(B, T_q, H, D).transpose(1, 2)  # (B, H, T_q, D)
        k = self.k_net(context).view(B, T_c, H, D).transpose(1, 2)  # (B, H, T_c, D)
        v = self.v_net(context).view(B, T_c, H, D).transpose(1, 2)  # (B, H, T_c, D)

        # Scaled dot-product attention
        weights = q @ k.transpose(-2, -1) / math.sqrt(D)  # (B, H, T_q, T_c)
        normalized_weights = F.softmax(weights, dim=-1)  # Normalize across context dimension
        attention = self.att_drop(normalized_weights @ v)  # (B, H, T_q, D)

        # Merge heads and project output
        attention = attention.transpose(1, 2).contiguous().view(B, T_q, H * D)  # (B, T_q, H*D)
        out = self.proj_drop(self.proj_net(attention))  # (B, T_q, out_dim)
        return out


class Memory:
    def __init__(self, max_seq_len, exps=None):
        self.keys = [
            "states", "actions", "probs", "rewards",
            "states_prime", "dones",
            "a_lsts", "values"
        ]
        self.max_seq_len = max_seq_len
        self.h0 = None
        self.h1 = None
        self.score = 0.0

        # used when copying other memory
        self.init_exps() if exps is None else exps

    def store(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.exps:
                self.exps[key].append(value)
            else:
                raise KeyError(f"Invalid key '{key}' provided to store.")

    def set_hidden(self, h):
        if self.h0 is None:
            self.h0 = h
        elif self.h1 is None:
            self.h1 = h

    def init_exps(self):
        self.exps = {key: [] for key in self.keys}

    def get_current_size(self):
        return len(next(iter(self.exps.values()), []))


def merge_dict(*dicts):
    result = {}
    for d in dicts:
        for key, value in d.items():
            if key in result:
                raise KeyError(f"Duplicate key found: {key}")
            result[key] = value
    return result


def check_saves_exist(log_dir: str = None, saved_folder: str = None):
    if log_dir is not None and os.path.exists(log_dir):
        clear_dir(log_dir)

    if saved_folder is not None and not os.path.exists(saved_folder):
        os.makedirs(saved_folder)

def check_record_exist(record_dir: str = None):
    if record_dir is not None:
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)
        else:
            clear_dir(record_dir)


def clear_dir(dir: str):
    try:
        user_input = input(
            f"> Found existing directory - \"{dir}\" \n" + \
            "> Clear and continue? (y/[n]): "
        ).strip().lower()
        if user_input == 'y':
            shutil.rmtree(dir)
            print("> Directory is cleared.\n")
            os.makedirs(dir)
        else:
            raise SystemExit("> The program has stopped.")
    except SystemExit as e:
        print(e)
        sys.exit()

action_data_sample = [[0], [1], [0], [1], [0], [0], [1], [1], [1], [0], [0], [1], [0], [1], [1], [0], [0], [0], [1], [1], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [0], [1], [1], [0], [0], [1], [0], [1], [0], [0], [1], [0], [0], [1], [0], [0], [0], [1], [1], [1], [1], [0], [1], [1], [0],
[0], [0], [0], [0], [1], [1], [1], [1], [0], [1], [1], [0], [1], [1], [0], [0], [0], [0], [1], [0], [1], [1], [1], [0], [1], [0], [0], [1], [0], [0], [0], [1], [1], [1], [1], [0], [1], [0], [0], [0], [0], [0], [1], [1], [0], [1], [1], [0], [1], [1], [1], [1], [1], [0], [1], [0], [1], [0], [0], [1], [0], [1], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0], [0], [1], [1], [1], [1], [1], [0], [1], [0], [1], [1], [1], [0], [1], [1], [0], [1], [0], [0], [1], [0], [0], [1], [1], [1], [0], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0], [1], [1], [1], [0], [1], [1], [0], [0], [0], [0], [0], [1], [1], [1], [0], [1], [0], [1], [0], [0], [0], [0], [0], [0], [0],
[1], [1], [1], [1], [1], [1], [0], [1], [1], [0], [1], [0], [1], [1], [0], [1], [0], [1], [0], [0], [1], [0], [1], [0], [1], [0], [0], [1], [1], [1], [1], [0], [0], [0], [1], [0], [0], [1], [1], [1], [1], [0], [1], [0], [0], [1], [0], [0], [0], [1], [1], [0], [1], [1], [0], [1], [1], [0], [1], [0], [0], [1], [1], [1], [1], [0], [0], [0], [0], [1], [1], [1], [0], [0], [0], [1], [1], [0], [1], [1], [0], [1], [1], [1], [0], [0], [0], [0], [1], [1], [0], [1], [1], [1], [1], [1], [1], [0], [1], [0], [1], [0], [0], [0], [0], [0], [1], [0], [1], [1], [0], [0], [1], [0], [1], [0], [0], [0], [0], [1], [0], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [0], [1], [0], [1], [1],
[1], [1], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [1], [1], [0], [1], [0], [0], [0], [0], [0], [1], [0], [0], [0], [1], [0], [1], [0], [1], [0], [1], [1], [1], [1], [0], [0], [1], [0], [1], [1], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [0], [0], [0], [1], [1], [1], [0], [0], [1], [1], [1], [0], [0], [0], [1], [0], [1], [0], [1], [1], [0], [0], [0], [0]]


def clamp(target, lower_bound, upper_bound):
    return max(lower_bound, min(target, upper_bound))