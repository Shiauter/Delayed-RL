import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

pi = torch.tensor([0.5, 0.5]).float()
c = Categorical(pi)
print(c.entropy())
print(-1 + 5 - 2)