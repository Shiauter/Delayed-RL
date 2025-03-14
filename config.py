from dataclasses import dataclass, field, asdict
import gym
import json
import torch

@dataclass
class Config:
    env_name: str
    model_root: str = "./models"
    experiment_name = "test_vrnn2"
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
    lr_pred_model: float = 3e-4
    lr_policy: float = 3e-4
    lr: float = 3e-4
    eps_clip: float = 0.2
    K_epoch_training: int = 500
    K_epoch_pred_model: int = 10
    K_epoch_policy: int = 3
    K_epoch_learn: int = 5
    delay: int = 4
    p_iters: int = delay
    num_actors: int = 10
    num_memos: int = 10
    T_horizon: int = 500
    hidden_size: int = 32
    batch_size: int = 50 # for predicting s_ti
    h0: list = field(init=False)
    # device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # for debugging
    do_save: bool = True
    do_train: bool = True

    def __post_init__(self):
        env = gym.make(self.env_name)
        self.s_size = env.observation_space.shape[0]
        self.a_size = env.action_space.n
        env.close()

        self.h0 = [1, 1, self.hidden_size]

    def get_json(self):
        config_dict = asdict(self)
        config_text = json.dumps(config_dict, indent=4)
        return config_text
