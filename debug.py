import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
# from util import Categorical

def test(*args):
    print(args)
    print(torch.cat(args, dim=-1))

a = torch.tensor([1,2])
b = torch.tensor([3,4])
test(b)