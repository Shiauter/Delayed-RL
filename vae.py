import torch
import torch.nn as nn
import torch.nn.functional as F
import math

EPS = 1e-6
LOG_2PI = math.log(2 * math.pi)

class VAE(nn.Module):
    def __init__(self, x_dim, z_dim, a_dim, h_dim, pred_s_source: str, nll_include_const: bool, set_std_to_1: bool):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.pred_s_source = pred_s_source
        self.nll_include_const = nll_include_const
        self.set_std_to_1 = set_std_to_1

        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.ReLU()
        )
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
            nn.ReLU()
        )

        self.enc_net = nn.Sequential(
            nn.Linear(x_dim + x_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Linear(h_dim, z_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(x_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Linear(h_dim, z_dim)

        self.dec_net = nn.Sequential(
            nn.Linear(z_dim + a_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.dec_std = nn.Linear(h_dim, x_dim)
        self.dec_mean = nn.Linear(h_dim, x_dim)

    def encode(self, *args):
        enc_t = self.enc_net(torch.cat(args, dim=-1))
        enc_mean_t, enc_std_t = self.enc_mean(enc_t), self.enc_std(enc_t)
        enc_std_t = torch.clamp(F.softplus(enc_std_t), min=EPS)
        if self.set_std_to_1:
            enc_std_t = torch.ones_like(enc_std_t)
        return enc_mean_t, enc_std_t

    def decode(self, *args):
        dec_t = self.dec_net(torch.cat(args, dim=-1))
        dec_mean_t, dec_std_t = self.dec_mean(dec_t), self.dec_std(dec_t)
        dec_std_t = torch.clamp(F.softplus(dec_std_t), min=EPS)
        if self.set_std_to_1:
            dec_std_t = torch.ones_like(dec_std_t)
        return dec_mean_t, dec_std_t

    def prior(self, *args):
        prior_t = self.prior_net(torch.cat(args, dim=-1))
        prior_mean_t, prior_std_t = self.prior_mean(prior_t), self.prior_std(prior_t)
        prior_std_t = torch.clamp(F.softplus(prior_std_t), min=EPS)
        if self.set_std_to_1:
            prior_std_t = torch.ones_like(prior_std_t)
        return prior_mean_t, prior_std_t

    # for training
    def reconstruct(self, x, s_t, a, h):
        phi_x_t = self.phi_x(x)
        enc_mean_t, enc_std_t = self.encode(phi_x_t, s_t, h)
        phi_z_t = self.phi_z(self._reparameterized_sample(enc_mean_t, enc_std_t))

        prior_mean_t, prior_std_t = self.prior(s_t, h)

        dec_mean_t, dec_std_t = self.decode(phi_z_t, a, h)
        if self.pred_s_source == "dec_mean_t": pred_s = dec_mean_t
        elif self.pred_s_source == "sampled_s": pred_s = self._reparameterized_sample(dec_mean_t, dec_std_t)

        kld_loss = self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
        nll_loss = self._nll_gauss(dec_mean_t, dec_std_t, x)
        # print("####")
        # print("nll:", nll_loss.mean(dim=-1))
        mse_loss = torch.pow(x - dec_mean_t, 2).sum(dim=-1)
        # print("mse sum feature", mse_loss.mean(dim=-1))
        # mse_loss = F.mse_loss(x, dec_mean_t, reduction="sum")
        # print("mse sum", mse_loss)
        # mse_loss = F.mse_loss(x, dec_mean_t, reduction="mean")
        # print("mse mean", mse_loss)
        # print("$$$$")
        return kld_loss, nll_loss, phi_x_t, phi_z_t, mse_loss, pred_s, dec_std_t

    def forward(self, s_t, a, h):
        # s = (batch, s_size)
        # a = (batch, 1)
        # h = (batch, hidden_size)
        # print("s, a, h ->", s.shape, a.shape, h.shape)
        prior_mean_t, prior_std_t = self.prior(s_t, h)
        # print("prior mean & std ->", prior_mean_t.shape, prior_std_t.shape)
        z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
        # print("z ->", z_t.shape)
        phi_z_t = self.phi_z(z_t)
        # print("phi_z_t ->", phi_z_t.shape)
        # print("dec in ->", torch.cat([phi_z_t, a, h], dim=-1).shape)
        dec_mean_t, dec_std_t = self.decode(phi_z_t, a, h)
        if self.pred_s_source == "dec_mean_t": pred_s = dec_mean_t
        elif self.pred_s_source == "sampled_s": pred_s = self._reparameterized_sample(dec_mean_t, dec_std_t)

        # print("dec_mean_t ->", dec_mean_t.shape)
        phi_x_t = self.phi_x(dec_mean_t)
        # print("phi_x_t ->", phi_x_t.shape)

        return pred_s, phi_x_t, phi_z_t

    def _reparameterized_sample(self, mean, std):
        eps = torch.randn_like(std)
        return mean + eps * torch.clamp(std, min=EPS)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        kld_element = (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) + \
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / (std_2.pow(2) + EPS) - 1) * 0.5
        return	torch.sum(kld_element, dim=-1)

    def _nll_gauss(self, mean, std, x):
        std = torch.clamp(std, min=EPS)
        var = std.pow(2)
        constant = 2 * torch.pi if self.nll_include_const else 1
        log_term = torch.log(var * constant)
        sqr_term = (x - mean).pow(2) / var
        nll_element = (log_term + sqr_term) / 2
        nll_loss = torch.sum(nll_element, dim=-1)
        pytorch_nll_loss_sum = F.gaussian_nll_loss(mean, x, var, reduction="sum")
        pytorch_nll_loss_mean = F.gaussian_nll_loss(mean, x, var)
        return nll_loss
