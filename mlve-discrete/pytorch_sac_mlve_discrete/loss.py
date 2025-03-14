from __future__ import print_function

import torch
import torch.nn as nn


class ContrasLoss(nn.Module):
    """Based on Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, t=0.005):
        super(ContrasLoss, self).__init__()
        self.temperature = temperature
        self.t = t

    def forward(self, states, actions, V_value, device):
        V_value = torch.sigmoid(V_value)
        # phi_i,j
        bs, n = states.shape
        phi_ij = torch.matmul(actions, torch.ones_like(actions).T) - torch.matmul(torch.ones_like(actions), actions.T) + 1e-8
        phi_ij = torch.exp(-(phi_ij ** 2))
        phi_ij = torch.sigmoid(phi_ij) * self.t + self.temperature

        s_ij = torch.matmul(states, states.T)

        diff_a = torch.abs(actions - actions.T)
        diff_V = torch.abs(V_value - V_value.T)
        mask = (diff_a < 0.5) * (diff_V < 0.5)
        logits_mask = torch.scatter(
            torch.ones_like(s_ij),
            1,
            torch.arange(bs).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        logits = torch.div(s_ij, phi_ij)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # loss
        loss = - mean_log_prob_pos.mean()

        return loss