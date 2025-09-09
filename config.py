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
    reward_scale: float = 1e-2

    # policy
    gamma: float = 0.99
    lmbda: float = 0.95
    critic_weight: float = 0.7
    entropy_weight: float = 5e-3
    advtg_norm: bool = False
    eps_clip: float = 0.1
    eps_clip_value: float = 0.1
    policy_dropout: float = 0.0

    # pred_model
    p_iters: int = delay
    z_size: int = 16
    reconst_loss_method: str = "NLL" # NLL, MSE
    pred_s_source: str = "dec_mean_t" # sampled_s, dec_mean_t
    nll_include_const: bool = True # only for nll
    pause_update_ep: int = None # only for separate learning
    set_std_to_1: bool = False
    z_source: str = "mean" # mean, sampled
    joint_elbo_weight: float = 0.5
    rollout_loss_weight: float = 1.0

    # training params
    learning_mode: str = "separate" # separate, joint
    K_epoch_training: int = 300
    epoch_joint: int = 5
    epoch_pred_model: int = 3
    epoch_policy: int = 5
    lr_joint: float = 1e-3
    lr_pred_model: float = 1e-3
    lr_policy: float = 1e-3
    num_actors: int = 10
    num_memos: int = 10
    batch_size: int = 50 # for predicting s_ti
    do_lr_sched: bool = False
    do_draw_graph: bool = False
    device: str = "cpu" # bug: GPU is slower than CPU
    do_save: bool = True
    do_train: bool = True

    # S/L
    model_root: str = "./models"
    experiment_name = f"{reconst_loss_method}_{pred_s_source}_delay_{delay}_{learning_mode}"
    model_name: str = "action_delay.tar"
    log_root: str = "./logs" # used in tensorboard
    log_dir = f"{log_root}/meeting_2025_09_05/vrnn_v2_baseline4/{experiment_name}"
    saved_folder = f"{model_root}/{experiment_name}"
    record_dir =f"{saved_folder}/records"
    record_interval: int = 10 # every n epoch


    def __post_init__(self):
        env = gym.make(self.env_name)
        self.s_size = env.observation_space.shape[0]
        self.a_size = env.action_space.n
        env.close()

        self.h0 = [1, 1, self.hidden_size]
        self.p_iters = self.delay

    def get_json(self):
        config_dict = asdict(self)
        config_text = json.dumps(config_dict, indent=4)
        return config_text
