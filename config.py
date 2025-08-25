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
    delay: int = 0
    hidden_size: int = 32
    h0: list = field(init=False)
    T_horizon: int = 500
    reward_scale: float = 1.0

    # policy
    gamma: float = 0.99
    lmbda: float = 0.95
    critic_weight: float = 0.7
    entropy_weight: float = 0.005
    advtg_norm: bool = True
    gate_reg_weight: float = 0.0
    gate_reg_weight_to_set: float = 0.0
    set_gate_reg_weight_at_ep: int = 0
    eps_clip: float = 0.2
    policy_dropout: float = 0.0
    # kld_policy_range: list = field(init=False)

    # pred_model
    p_iters: int = delay
    z_size: int = 16
    reconst_loss_method: str = "NLL" # NLL, MSE
    pred_s_source: str = "dec_mean_t" # sampled_s, dec_mean_t
    nll_include_const: bool = True # only for nll
    pause_update_ep: int = None # only for separate learning
    set_std_to_1: bool = False
    z_source: str = "sampled" # mean, sampled
    # kld_range: list = field(init=False)
    joint_elbo_weight: float = 0.5

    # training params
    learning_mode: str = "separate" # separate, joint
    # lr_pred_model: float = field(init=False)
    # lr_policy: float = field(init=False)
    # lr: float = field(init=False)
    K_epoch_training: int = 300
    epoch_joint: int = 5
    epoch_pred_model: int = 5
    epoch_policy: int = 5
    lr_joint: float = 3e-4
    lr_pred_model: float = 3e-4
    lr_policy: float = 3e-4
    # epoch_tier_joint: int = 2
    # epoch_tier_policy: int = 2
    # epoch_tier_pred_model: int = 2
    # lr_tier_joint: int = 2
    # lr_tier_policy: int = 2
    # lr_tier_pred_model: int = 2
    num_actors: int = 10
    num_memos: int = 10
    batch_size: int = 50 # for predicting s_ti
    # epoch_tier: list = field(init=False)
    # lr_tier: list = field(init=False)
    do_lr_sched: bool = True
    device: str = "cpu" # bug: GPU is slower than CPU
    do_save: bool = True
    do_train: bool = True

    # S/L
    model_root: str = "./models"
    experiment_name = f"{reconst_loss_method}_{pred_s_source}_delay_{delay}_{learning_mode}"
    model_name: str = "action_delay.tar"
    log_root: str = "./logs" # used in tensorboard
    log_dir = f"{log_root}/meeting_2025_08_29/vrnn_v2_add_scheduler2/{experiment_name}"
    saved_folder = f"{model_root}/{experiment_name}"
    record_dir =f"{saved_folder}/records"
    record_interval: int = 10 # every n epoch


    def __post_init__(self):
        env = gym.make(self.env_name)
        self.s_size = env.observation_space.shape[0]
        self.a_size = env.action_space.n
        env.close()

        self.h0 = [1, 1, self.hidden_size]

        # self.epoch_tier = [1, 3, 5, 7, 9]
        # self.lr_tier = [1e-3, 5e-4, 3e-4, 1e-4, 5e-5]

        # # format -> from small to large
        # self.kld_policy_range = [5e-3, 3e-2]
        # self.kld_range = [0.3, 1.2]

    def get_json(self):
        config_dict = asdict(self)
        config_text = json.dumps(config_dict, indent=4)
        return config_text
