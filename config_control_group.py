from dataclasses import dataclass, field, asdict
import gym
import json

@dataclass
class Config:
    # env
    env_name: str = "CartPole-v1"
    env_seed: int = None
    s_size: int = field(init=False)
    a_size: int = field(init=False)
    delay: int = field(init=False)
    hidden_size: int = 64
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
    pause_update_ep: int = None # only for separate learning

    # training params
    used_method: str = None # PPO, LSTM_PPO, P_Model
    learning_mode: str = "separate" # separate, joint
    K_epoch_training: int = 300
    epoch_joint: int = 5
    epoch_pred_model: int = 5
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
    do_record: bool = False

    # S/L
    model_root: str = "./models"
    experiment_name: str = field(init=False)
    model_name: str = "action_delay.tar"
    log_root: str = "./logs" # used in tensorboard
    log_dir: str = field(init=False)
    saved_folder: str = field(init=False)
    record_dir: str = field(init=False)
    record_interval: int = 10 # every n epoch


    def __post_init__(self):
        env = gym.make(self.env_name)
        self.s_size = env.observation_space.shape[0]
        self.a_size = env.action_space.n
        env.close()

        self.h0 = [1, 1, self.hidden_size]
        self.p_iters = self.delay

    def _refresh_path_args(self):
        self.experiment_name = f"{self.used_method}_delay_{self.delay}_{self.learning_mode}"
        self.log_dir = f"{self.log_root}/meeting_2025_10_03/control_group_baseline/{self.experiment_name}"
        self.saved_folder = f"{self.model_root}/{self.experiment_name}"
        self.record_dir =f"{self.saved_folder}/records"

    def apply_args_override(self, delay: int=None, used_method: str=None):
        if delay is not None:
            self.delay = delay
            self.p_iters = delay
        else:
            raise ValueError(f"Unsupported delay: {delay}")

        if used_method in {"PPO", "LSTM_PPO", "P_Model"}:
            self.used_method = used_method
        else:
            raise KeyError(f"Unknown used_method: {used_method}")
        self._refresh_path_args()

    def get_json(self):
        config_dict = asdict(self)
        config_text = json.dumps(config_dict, indent=4)
        return config_text
