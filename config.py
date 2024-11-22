from dataclasses import dataclass, field
import torch, gym

@dataclass
class Config:
    env_name: str
    s_size: int = field(init=False)
    a_size: int = field(init=False)
    gamma: float = 0.99
    lmbda: float = 0.95
    eps_clip: float = 0.1
    K_epoch_policy: int = 3
    K_epoch_pred_model: int = 10
    delay: int = 4
    p_iters: int = 4
    num_memos: int = 10
    num_actors: int = 5
    T_horizon: int = 500
    h0: tuple = (
        torch.zeros([1, 1, 64], dtype=torch.float),
        torch.zeros([1, 1, 64], dtype=torch.float),
    )

    def __post_init__(self):
        env = gym.make(self.env_name)
        self.s_size = env.observation_space.shape[0]
        self.a_size = env.action_space.n
        env.close()