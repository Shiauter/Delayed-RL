import torch
from torch.distributions.utils import probs_to_logits
import os, sys, shutil

class Categorical:
    def __init__(self, probs_shape):
        # NOTE: probs_shape is supposed to be
        #       the shape of probs that will be
        #       produced by policy network
        if len(probs_shape) < 1:
            raise ValueError("`probs_shape` must be at least 1.")
        self.probs_dim = len(probs_shape)
        self.probs_shape = probs_shape
        self._num_events = probs_shape[-1]
        self._batch_shape = probs_shape[:-1] if self.probs_dim > 1 else torch.Size()
        self._event_shape=torch.Size()

    def set_probs(self, probs):
        # normalized the probs
        self.probs = probs / probs.sum(-1, keepdim=True)
        # log probabilities
        # domain range changed from [0, 1] -> [-inf, inf]
        self.logits = probs_to_logits(self.probs)

    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        # reshape the probs to 2D
        probs_2d = self.probs.reshape(-1, self._num_events)
        # for each row, return n results with replacement, n == 1 result in my case
        samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
        # reshape the results to specified shape
        return samples_2d.reshape(sample_shape + self._batch_shape + self._event_shape)

    def log_prob(self, value):
        value = value.long().unsqueeze(-1)
        # make value and logits have matched shape
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        # for each row, return log_pmf[value[row]]
        return log_pmf.gather(-1, value).squeeze(-1)

    def entropy(self):
        # to avoid large negative log probability when log(0) occurred
        # we use "eps" instead of "min" here
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        # entropy
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)

class Memory:
    def __init__(self, max_seq_len, exps=None):
        self.keys = [
            "states", "actions", "probs", "rewards",
            "states_prime", "dones",
            "timesteps", "a_lsts"
        ]
        self.max_seq_len = max_seq_len
        self.score = 0.0

        # used when copying other memory
        self.init_exps() if exps is None else exps

    def store(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.exps:
                self.exps[key].append(value)
            else:
                raise KeyError(f"Invalid key '{key}' provided to store.")

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

def check_dir_exist(log_dir: str, saved_folder: str):
    if os.path.exists(log_dir):
        try:
            user_input = input(
                "> Found existing log directory. \n" + \
                "> Clear and continue？ (y/[n]): "
            ).strip().lower()
            if user_input == 'y':
                shutil.rmtree(log_dir)
                print("> Log directory is cleared.")
            else:
                raise SystemExit("> The program has stopped.")
        except SystemExit as e:
            print(e)
            sys.exit()

    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)