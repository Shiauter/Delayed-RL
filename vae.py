import torch
import torch.nn as nn
import math

EPS = torch.finfo(torch.float).eps # numerical logs

class VAE(nn.Module):
    def __init__(self, x_dim, z_dim, a_dim, h_dim):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.a_dim = a_dim
        self.h_dim = h_dim

        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.ReLU()
        )
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU()
        )

        self.enc_net = nn.Sequential(
            nn.Linear(x_dim + x_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus()
        )

        self.prior_net = nn.Sequential(
            nn.Linear(x_dim + h_dim, h_dim),
            nn.ReLU()
        )
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus()
        )

        self.dec_net = nn.Sequential(
            nn.Linear(z_dim + a_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus()
        )
        self.dec_mean = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid()
        )

        # projection
        self.proj = nn.Linear(x_dim + z_dim + a_dim, x_dim)

    def encode(self, *args):
        enc_t = self.enc_net(torch.cat(args, dim=-1))
        enc_mean_t, enc_std_t = self.enc_mean(enc_t), self.enc_std(enc_t)
        return enc_mean_t, enc_std_t

    def decode(self, *args):
        dec_t = self.dec_net(torch.cat(args, dim=-1))
        dec_mean_t, dec_std_t = self.dec_mean(dec_t), self.dec_std(dec_t)
        return dec_mean_t, dec_std_t

    def prior(self, *args):
        prior_t = self.prior_net(torch.cat(args, dim=-1))
        prior_mean_t, prior_std_t = self.prior_mean(prior_t), self.prior_std(prior_t)
        return prior_mean_t, prior_std_t

    # for training
    def reconstruct(self, x_truth, x_cond, h_truth, h_cond, a):
        phi_x_truth = self.phi_x(x_truth)
        enc_mean_t, enc_std_t = self.encode(phi_x_truth, x_cond, h_truth)
        phi_z_truth = self.phi_z((self._reparameterized_sample(enc_mean_t, enc_std_t)))

        prior_mean_t, prior_std_t = self.prior(x_cond, h_cond)
        phi_z_cond = self.phi_z((self._reparameterized_sample(prior_mean_t, prior_std_t)))

        dec_mean_truth, dec_std_truth = self.decode(phi_z_truth, a, h_truth)
        dec_mean_cond, _ = self.decode(phi_z_cond, a, h_cond)
        phi_x_cond = self.phi_x(dec_mean_cond)

        kld_loss = self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
        nll_loss = self._nll_gauss(dec_mean_truth, dec_std_truth, x_truth)

        return (kld_loss, nll_loss), \
            (
                phi_x_truth, phi_z_truth,
                phi_x_cond, phi_z_cond
            )

    def forward(self, s, a, h):
        # s = (batch, s_size)
        # a = (batch, 1)
        # h = (batch, hidden_size)
        # print("s, a, h ->", s.shape, a.shape, h.shape)
        prior_mean_t, prior_std_t = self.prior(s, h)
        # print("prior mean & std ->", prior_mean_t.shape, prior_std_t.shape)
        z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
        # print("z ->", z_t.shape)
        phi_z_t = self.phi_z(z_t)
        # print("phi_z_t ->", phi_z_t.shape)
        # print("dec in ->", torch.cat([phi_z_t, a, h], dim=-1).shape)
        dec_mean_t, _ = self.decode(phi_z_t, a, h)
        # print("dec_mean_t ->", dec_mean_t.shape)
        phi_x_t = self.phi_x(dec_mean_t)
        # print("phi_x_t ->", phi_x_t.shape)

        return dec_mean_t, phi_x_t, z_t, phi_z_t

    def _reparameterized_sample(self, mean, std):
        eps = torch.empty(size=std.size(), dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) +
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)

    def _nll_gauss(self, mean, std, x):
        return torch.sum(torch.log(std + 1e-8) + (x - mean).pow(2) / (2 * std.pow(2)) + 0.5 * torch.log(torch.tensor(2 * math.pi)))
