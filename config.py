from dataclasses import dataclass, field, asdict
import gym
import json
import torch

@dataclass
class Config:
    env_name: str
    model_root: str = "./models"
    experiment_name = "test_vrnn_with_fixed_data7"
    model_name: str = "action_delay.tar"
    log_root: str = "./logs" # used in tensorboard
    log_dir = f"{log_root}/{experiment_name}"
    saved_folder = f"{model_root}/{experiment_name}"
    record_dir =f"{saved_folder}/records"
    record_interval: int = 10 # every n epoch

    s_size: int = field(init=False)
    a_size: int = field(init=False)
    gamma: float = 0.99
    lmbda: float = 0.95
    critic_weight: float = 0.9
    entropy_weight: float = 0.001
    lr_pred_model: float = field(init=False)
    lr_policy: float = field(init=False)
    lr: float = field(init=False)
    eps_clip: float = 0.2
    K_epoch_training: int = 300
    K_epoch_pred_model: int = field(init=False)
    K_epoch_policy: int = field(init=False)
    K_epoch_learn: int = field(init=False)
    delay: int = 4
    p_iters: int = delay
    num_actors: int = 10
    num_memos: int = 10
    T_horizon: int = 500
    hidden_size: int = 32
    z_size: int = hidden_size // 2
    batch_size: int = 50 # for predicting s_ti
    h0: list = field(init=False)
    epoch_tier: list = field(init=False)
    lr_tier: list = field(init=False)
    # device: str = "cuda" if torch.cuda.is_available() else "cpu" # bug: GPU is slower than CPU
    device: str = "cpu"

    # for debugging
    do_save: bool = True
    do_train: bool = True

    def __post_init__(self):
        env = gym.make(self.env_name)
        self.s_size = env.observation_space.shape[0]
        self.a_size = env.action_space.n
        env.close()

        self.h0 = [1, 1, self.hidden_size]

        self.epoch_tier = [1, 3, 5, 7, 10]
        self.lr_tier = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
        init_tier = 2
        self.lr, self.lr_policy, self.lr_pred_model = self.lr_tier[init_tier], self.lr_tier[init_tier], self.lr_tier[init_tier]
        self.K_epoch_learn, self.K_epoch_policy, self.K_epoch_pred_model = self.epoch_tier[init_tier], self.epoch_tier[init_tier], self.epoch_tier[init_tier]

    def get_json(self):
        config_dict = asdict(self)
        config_text = json.dumps(config_dict, indent=4)
        return config_text
