from dataclasses import dataclass, field, asdict
import gym
import json

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
    reward_scale: float = 10.0

    # policy
    gamma: float = 0.99
    lmbda: float = 0.95
    critic_weight: float = 0.7
    entropy_weight: float = 0.001
    advtg_norm: bool = True
    gate_reg_weight: float = 0.0
    gate_reg_weight_to_set: float = 0.0
    set_gate_reg_weight_at_ep: int = 0
    eps_clip: float = 0.1
    policy_dropout: float = 0.1

    # pred_model
    p_iters: int = delay
    z_size: int = 16
    reconst_loss_method: str = "NLL" # NLL, MSE
    pred_s_source: str = "dec_mean_t" # sampled_s, dec_mean_t
    nll_include_const: bool = True # only for nll
    pause_update_ep: int = None # only for separate learning
    set_std_to_1: bool = False

    # training params
    learning_mode: str = "joint" # separate, joint
    lr_pred_model: float = field(init=False)
    lr_policy: float = field(init=False)
    lr: float = field(init=False)
    K_epoch_training: int = 1
    K_epoch_pred_model: int = field(init=False)
    K_epoch_policy: int = field(init=False)
    K_epoch_learn: int = field(init=False)
    num_actors: int = 10
    num_memos: int = 1
    batch_size: int = 50 # for predicting s_ti
    epoch_tier: list = field(init=False)
    lr_tier: list = field(init=False)
    device: str = "cpu" # bug: GPU is slower than CPU
    do_save: bool = False
    do_train: bool = True

    # S/L
    model_root: str = "./models"
    experiment_name = f"{reconst_loss_method}_{pred_s_source}_delay_{delay}_{learning_mode}"
    model_name: str = "action_delay.tar"
    log_root: str = "./logs" # used in tensorboard
    log_dir = f"{log_root}/{experiment_name}"
    saved_folder = f"{model_root}/{experiment_name}"
    record_dir =f"{saved_folder}/records"
    record_interval: int = 10 # every n epoch


    def __post_init__(self):
        env = gym.make(self.env_name)
        self.s_size = env.observation_space.shape[0]
        self.a_size = env.action_space.n
        env.close()

        self.h0 = [1, 1, self.hidden_size]

        self.epoch_tier = [10, 15, 20, 25, 30]
        self.lr_tier = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
        init_tier = 2
        self.lr, self.lr_policy, self.lr_pred_model = self.lr_tier[init_tier], self.lr_tier[init_tier], self.lr_tier[init_tier]
        self.K_epoch_learn, self.K_epoch_policy, self.K_epoch_pred_model = self.epoch_tier[init_tier], self.epoch_tier[init_tier], self.epoch_tier[init_tier]

    def get_json(self):
        config_dict = asdict(self)
        config_text = json.dumps(config_dict, indent=4)
        return config_text
