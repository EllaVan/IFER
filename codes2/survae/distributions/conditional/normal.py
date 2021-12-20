import math
import torch
import torch.nn as nn
from torch.distributions import Normal, HalfNormal
from survae.distributions.conditional.base import ConditionalDistribution
from survae.utils.tensors import sum_except_batch

class ConditionalNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and log_std."""

    def __init__(self, net, split_dim=-1, relu_mu=False):
        super(ConditionalNormal, self).__init__()
        self.net = net
        self.split_dim = split_dim
        self.relu_mu = relu_mu

    def cond_dist(self, context, feedback=None):
        if feedback is None:
            params = self.net(context)
        else:
            params = self.net(context, feedback=feedback)
        mean, log_std = torch.chunk(params, chunks=2, dim=self.split_dim)
        log_std = log_std.exp()
        log_std = log_std + torch.max(log_std)
        if self.relu_mu:
            mean = nn.ReLU()(mean)
        mean = mean + torch.max(mean)
        return Normal(loc=mean, scale=log_std)
            # try:
            #     ret = Normal(loc=mean, scale=log_std.exp())
            # except ValueError as e:
            #     print("log_std: ", log_std)
            #     print(log_std.type)
            # else:
            #     ret = Normal(loc=mean, scale=log_std.exp())
            #     return Normal(loc=mean, scale=log_std.exp())

    def log_prob(self, x, context, feedback=None):
        dist = self.cond_dist(context, feedback)
        return sum_except_batch(dist.log_prob(x))

    def sample(self, context, feedback=None):
        dist = self.cond_dist(context, feedback)
        return dist.rsample()

    def sample_with_log_prob(self, context, feedback=None):
        dist = self.cond_dist(context, feedback=feedback)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z, log_prob

    def mean(self, context, feedback=None):
        return self.cond_dist(context, feedback=feedback).mean

    def mean_stddev(self, context):
        dist = self.cond_dist(context)
        return dist.mean, dist.stddev