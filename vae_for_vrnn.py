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
            nn.Linear(x_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Linear(h_dim, z_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(a_dim + h_dim, h_dim),
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

    def enc(self, phi_x, h):
        return self._guassian_head(
            self.enc_net, self.enc_mean, self.enc_std,
            phi_x, h
        )

    def dec(self, phi_z, a, h):
        return self._guassian_head(
            self.dec_net, self.dec_mean, self.dec_std,
            phi_z, a, h
        )

    def prior(self, a, h):
        return self._guassian_head(
            self.prior_net, self.prior_mean, self.prior_std,
            a, h
        )

    def cal_loss(self, x, a, h):
        phi_x = self.phi_x(x) # truth state
        enc_mean, enc_std = self.enc(phi_x, h)
        z_t = self._reparameterized_sample(enc_mean, enc_std)
        phi_z = self.phi_z(z_t)
        dec_mean, dec_std = self.dec(phi_z, a, h)

        prior_mean_t, prior_std_t = self.prior(a, h) # pred state

        kld_loss = self._kld_gauss(enc_mean, enc_std, prior_mean_t, prior_std_t)
        nll_loss = self._nll_gauss(dec_mean, dec_std, x)
        mse_loss = torch.pow(x - dec_mean, 2).sum(dim=-1)
        return kld_loss, nll_loss, mse_loss, dec_std

    def forward(self, s_t, h):
        # s = (batch, s_size)
        # h = (batch, hidden_size)

        phi_x = self.phi_x(s_t)
        enc_mean, enc_std = self.enc(phi_x, h)
        z_t = self._reparameterized_sample(enc_mean, enc_std)
        phi_z = self.phi_z(z_t)

        return phi_x, phi_z

    def rollout(self, a, h):
        # a = (batch, 1)
        # h = (batch, hidden_size)

        prior_mean, prior_std = self.prior(a, h)
        z_t = self._reparameterized_sample(prior_mean, prior_std)
        phi_z = self.phi_z(z_t)

        dec_mean, dec_std = self.dec(phi_z, a, h)
        if self.pred_s_source == "dec_mean_t": pred_s = dec_mean
        elif self.pred_s_source == "sampled_s": pred_s = self._reparameterized_sample(dec_mean, dec_std)
        phi_x = self.phi_x(pred_s)

        return phi_x, phi_z

    def _guassian_head(self, net, mean_layer, std_layer, *inputs):
        x = net(torch.cat(inputs, dim=-1))
        x_mean, x_std = mean_layer(x), std_layer(x)
        x_std = torch.clamp(F.softplus(x_std), min=EPS)
        if self.set_std_to_1:
            x_std = torch.ones_like(x_std)
        return x_mean, x_std

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
        return nll_loss
