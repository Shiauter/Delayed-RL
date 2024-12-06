from dataclasses import dataclass, field, asdict
import gym
import json

@dataclass
class Config:
    env_name: str
    model_root: str = "./models"
    experiment_name = "fixing_small_critic_loss"
    model_name: str = "action_delay.tar"
    log_root: str = "./logs" # used in tensorboard
    s_size: int = field(init=False)
    a_size: int = field(init=False)
    gamma: float = 0.99
    lmbda: float = 0.95
    critic_weight: float =0.8
    entropy_weight: float = 0.01
    lr_pred_model: float = 3e-4
    lr_policy: float = 1e-3
    eps_clip: float = 0.2
    K_epoch_training: int = 250
    K_epoch_pred_model: int = 10
    K_epoch_policy: int = 5
    delay: int = 4
    p_iters: int = delay
    num_memos: int = 10
    num_actors: int = 5
    T_horizon: int = 500
    hidden_size: int = 64
    h0: list = field(init=False)

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
