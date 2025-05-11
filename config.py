from dataclasses import dataclass, field, asdict
import gym
import json
import torch

@dataclass
class Config:
    # env
    env_name: str
    env_seed: int = None
    s_size: int = field(init=False)
    a_size: int = field(init=False)
    delay: int = 1
    hidden_size: int = 32
    h0: list = field(init=False)
    T_horizon: int = 500

    # policy
    gamma: float = 0.99
    lmbda: float = 0.95
    critic_weight: float = 0.9
    entropy_weight: float = 0.001
    eps_clip: float = 0.2

    # pred_model
    p_iters: int = delay
    z_size: int = 16
    reconst_loss_method: str = "NLL" # NLL, MSE
    pred_s_source: str = "sampled_s" # sampled_s, dec_mean_t
    nll_include_const: bool = True # only for nll
    pause_update_ep: int = 100
    set_std_to_1: bool = False

    # training params
    lr_pred_model: float = field(init=False)
    lr_policy: float = field(init=False)
    lr: float = field(init=False)
    K_epoch_training: int = 500
    K_epoch_pred_model: int = field(init=False)
    K_epoch_policy: int = field(init=False)
    K_epoch_learn: int = field(init=False)
    num_actors: int = 10
    num_memos: int = 1
    batch_size: int = 50 # for predicting s_ti
    epoch_tier: list = field(init=False)
    lr_tier: list = field(init=False)
    device: str = "cpu" # bug: GPU is slower than CPU
    do_save: bool = True
    do_train: bool = True

    # S/L
    model_root: str = "./models"
    experiment_name = f"{reconst_loss_method}_{pred_s_source}_delay_{delay}_only_policy_loss"
    model_name: str = "action_delay.tar"
    log_root: str = "./logs" # used in tensorboard
    log_dir = f"{log_root}/testing_ppo_loss/{experiment_name}"
    saved_folder = f"{model_root}/{experiment_name}"
    record_dir =f"{saved_folder}/records"
    record_interval: int = 10 # every n epoch


    def __post_init__(self):
        env = gym.make(self.env_name)
        self.s_size = env.observation_space.shape[0]
        self.a_size = env.action_space.n
        env.close()

        self.h0 = [1, 1, self.hidden_size]

        self.epoch_tier = [1, 3, 5, 7, 10]
        self.lr_tier = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        init_tier = 2
        self.lr, self.lr_policy, self.lr_pred_model = self.lr_tier[init_tier], self.lr_tier[init_tier], self.lr_tier[init_tier]
        self.K_epoch_learn, self.K_epoch_policy, self.K_epoch_pred_model = self.epoch_tier[init_tier], self.epoch_tier[init_tier], self.epoch_tier[init_tier]

    def get_json(self):
        config_dict = asdict(self)
        config_text = json.dumps(config_dict, indent=4)
        return config_text
