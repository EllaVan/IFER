import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import tcvae.tcvae as tcvae
from tcvae.tcvae import logsumexp
import random
import numpy as np
import math

def WeightedL1(pred, gt):
    wt = (pred - gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0), wt.size(1))
    loss = wt * (pred - gt).abs()
    return loss.sum() / loss.size(0)

def calc_gradient_penalty(device, netD, real_data, fake_data, input_att, opt):
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())
    ones = ones.to(device)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

# TCVAE
def get_tcvae_kl_loss(mean, log_var, z, beta, opt):
    batch_size = mean.shape[0]
    z_dim = mean.shape[1]
    log_var *= 0.5      # this code base scales the output of parameter network
    prior_parameters = torch.zeros(batch_size, z_dim, 2).cuda()
    z_params = torch.cat((mean.unsqueeze(2), log_var.unsqueeze(2)), dim=2)
    prior_dist = tcvae.Normal()
    q_dist = tcvae.Normal()
    zs = z
    logpz = prior_dist.log_density(zs, params=prior_parameters).view(batch_size, -1).sum(1)
    logqz_condx = q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)
    _logqz = q_dist.log_density(
        zs.view(batch_size, 1, z_dim),
        z_params.view(1, batch_size, z_dim, q_dist.nparams)
    )
    logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * opt.ntrain)).sum(1)
    logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * opt.ntrain))
    kld = (logqz_condx - logqz) + beta * (logqz - logqz_prodmarginals) + (logqz_prodmarginals - logpz)
    return kld.mean()

def loss_fn(recon_x, x, mean, log_var, opt, z, beta=1.0):
    if opt.recon == "bce":
        BCE = torch.nn.functional.binary_cross_entropy(recon_x + 1e-12, x.detach(), reduction='sum')
        BCE = BCE.sum() / x.size(0)
    elif opt.recon == "l2":
        BCE = torch.sum(torch.pow(recon_x - x.detach(), 2), 1).mean()
    elif opt.recon == "l1":
        BCE = torch.sum(torch.abs(recon_x - x.detach()), 1).mean()
    if not opt.z_disentangle:
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    elif not opt.zd_tcvae:
        KLD = -0.5 * beta * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    else:
        KLD = get_tcvae_kl_loss(mean, log_var, z, beta, opt)
    return (BCE + KLD)

def mse_loss(X, x_recon):
    # x_recon = model.conditional_sample(X, S, deterministic=deterministic)
    mse_loss_val = nn.MSELoss()(x_recon, X)
    return mse_loss_val

def sample_with_gradient(device, mean, log_var):
    batch_size = mean.shape[0]
    latent_size = mean.shape[1]
    std = torch.exp(log_var)
    eps = torch.randn([batch_size, latent_size]).to(device)
    eps = Variable(eps.to(device))
    z = eps * std + mean  # torch.Size([64, 312])
    return z

def conditional_sample(device, netE, netF, netG, netDec, x, y, opt, deterministic=False, z0=False):
    # x is feature vector
    # y is attribute vector
    if not opt.survae:
        means, log_var = netE(x, y)
        if deterministic:
            z = means
        else:
            # z = torch.normal(means, torch.exp(0.5 * log_var))
            z = sample_with_gradient(device, means, 0.5 * log_var)
    else:
        z, _ = netE(x, y)
    if z0:
        z = torch.zeros(z.shape).to(device)
    x_gen = netG(z, c=y)
    if netF is not None:
        _ = netDec(x_gen)
        dec_hidden_feat = netDec.getLayersOutDet()  # no detach layers
        feedback_out = netF(dec_hidden_feat)
        x_gen = netG(z, a1=opt.a2, c=y, feedback_layers=feedback_out)
    return x_gen

def contrastive_loss(device, netE, netF, netG, netDec, seen_clf, batch_x, batch_l, data,
                     seenSemanticVectors, opt, deterministic=False, temperature=1.0, K=30):
    '''
    # V1: positive samples are batch_x. negative samples are generated using test classes
    batch_size = batch_x.shape[0]
    x_dim = batch_x.shape[1]
    train_classes = data.seenclasses.numpy().tolist()
    n_samples = 1
    cf_x, _ = generate_syn_feature_cf(netE, netF, netG, netDec, batch_x, data.unseenclasses, data, opt, n_samples=n_samples, deterministic=deterministic)
    batch_l_in_seen = np.array([train_classes.index(item) for item in batch_l])
    batch_l_in_seen = torch.from_numpy(batch_l_in_seen).cuda()
    cf_x = cf_x.view(-1, x_dim)
    cf_logits = seen_clf(cf_x).view(batch_size, -1, data.ntrain_class)
    batch_logits = seen_clf(batch_x)
    gt_logits = torch.gather(batch_logits, 1, batch_l_in_seen.unsqueeze(1))
    t = batch_l_in_seen.unsqueeze(1).expand(-1, 50).unsqueeze(2)
    cf_logits_at_gt = torch.gather(cf_logits, 2, t).squeeze(2)
    contra_logits = torch.cat((gt_logits, cf_logits_at_gt), dim=1)
    label = torch.zeros(batch_size).cuda().long()
    contrastive_loss = nn.CrossEntropyLoss()(contra_logits / temperature, label)
    return contrastive_loss
    '''
    # V2: positive samples are reconstructed batch_x. negative samples are generated using other seenclasses
    batch_size = batch_x.shape[0]
    x_dim = batch_x.shape[1]
    s_dim = opt.attSize
    batch_s = seenSemanticVectors[batch_l]
    train_classes = list(range(data.dataset.seenclasses))
    # gen_classes = data.seenclasses.numpy().tolist() + data.unseenclasses.numpy().tolist()
    gen_classes = train_classes
    attributes = Variable(torch.zeros((batch_size, K + 1, s_dim))).to(device)
    x = Variable(torch.zeros((batch_size, K + 1, x_dim)), requires_grad=True).to(device)
    for i in range(batch_size):
        # First element is ground-truth
        attributes[i][0] = batch_s[i]
        # Negative samples
        negative_classes = [item for item in gen_classes if item != batch_l[i]]
        negative_classes_selected = random.sample(negative_classes, K)
        negative_attributes = seenSemanticVectors[negative_classes_selected, :]
        attributes[i, 1:] = negative_attributes
        x[i] = batch_x[i].unsqueeze(0).expand(K + 1, -1)
    xv = x.view(batch_size * (K + 1), -1)
    yv = attributes.view(batch_size * (K + 1), -1)
    dictionary = conditional_sample(device, netE, netF, netG, netDec, xv, yv, opt, deterministic=deterministic)

    # V3: using euclidean distance
    if opt.contra_v == 4 or opt.contra_v == 3:
        dictionary = dictionary.view(batch_size, K + 1, x_dim)
        batch_x_expanded = batch_x.unsqueeze(1).expand(batch_size, K + 1, -1)
        neg_dist = -((batch_x_expanded - dictionary) ** 2).mean(dim=2) * temperature  # N*(K+1)
        label = torch.zeros(batch_size).to(device).long()
        contrastive_loss_euclidean = nn.CrossEntropyLoss()(neg_dist, label)

    # V2: using pretrain classifier
    if opt.contra_v == 4 or opt.contra_v == 2:
        batch_l_in_seen = np.array([train_classes.index(item) for item in batch_l])
        batch_l_in_seen = torch.from_numpy(batch_l_in_seen).cuda()
        t = batch_l_in_seen.unsqueeze(1).expand(-1, K + 1).unsqueeze(2)
        logits = seen_clf(dictionary).view(batch_size, K + 1, -1)
        logits_at_gt = torch.gather(logits, 2, t).squeeze(2) * temperature  # batch_size * (K+1)
        label = torch.zeros(batch_size).cuda().long()
        contrastive_loss_clf = nn.CrossEntropyLoss()(logits_at_gt, label)

    if opt.contra_v == 2:
        return contrastive_loss_clf
    if opt.contra_v == 3:
        return contrastive_loss_euclidean
    if opt.contra_v == 4:
        # V4: using pretrain classifier as well as euclidean distance
        return contrastive_loss_clf + contrastive_loss_euclidean

# conditional_sample(netE, netF, netG, netDec, x, y, opt, deterministic=False, z0=False):
def get_p_loss(netE, netF, netG, netDec, x, y, prototypes, opt):
    gen_prototypes = conditional_sample(netE, netF, netG, netDec, x, y, opt, deterministic=False, z0=True)
    p_loss = nn.MSELoss()(gen_prototypes, prototypes)
    return p_loss

def sample_using_zprior(netF, netG, netDec, y, opt, deterministic=False):
    syn_noise = torch.zeros(y.shape[0], opt.latentSize).cuda()
    syn_noise.normal_(0, 1)
    x_gen = netG(syn_noise, c=y)
    if netF is not None:
        _ = netDec(x_gen)
        dec_hidden_feat = netDec.getLayersOutDet()  # no detach layers
        feedback_out = netF(dec_hidden_feat)
        x_gen = netG(syn_noise, a1=opt.a2, c=y, feedback_layers=feedback_out)
    return x_gen

def unconstrained_z_loss(data, seen_clf, generator, netF, netDec, opt):
    num = 1
    train_classes = data.seenclasses.numpy().tolist()
    x = sample_using_zprior(netF, generator, netDec, data.attribute[data.seenclasses], opt, deterministic=False)
    batch_l_in_seen = np.array([train_classes.index(item) for item in data.seenclasses])
    batch_l_in_seen = torch.from_numpy(batch_l_in_seen).cuda()
    logits = seen_clf(x)
    loss = nn.CrossEntropyLoss()(logits, batch_l_in_seen)
    return loss

def zy_disentangle_loss(encoder, encoder_opt, z, y):
    # use actual y
    encoder.train()
    encoder_opt.zero_grad()
    logpz_y = encoder.log_prob(z.detach(), y.detach())
    encoder_d_loss = -logpz_y.mean()
    encoder_d_loss.backward()
    encoder_opt.step()
    # Another forward since encoder was just updated
    encoder.eval()
    logpz_y = encoder.log_prob(z, y)    # Minimize pz_y
    generator_d_loss = logpz_y.mean()
    return generator_d_loss

def yz_disentangle_loss(device, encoder, encoder_opt, y, z, epoch, batch_label, data, opt):
    # use actual y
    for param in encoder.parameters():
        param.requires_grad = True
    encoder.train()
    encoder_opt.zero_grad()
    # seenclasses = data.seenclasses.numpy().tolist()
    # batch_label_softmax = [seenclasses.index(l) for l in batch_label]
    # batch_label_softmax = torch.from_numpy(np.array(batch_label_softmax)).to(device).long()
    batch_label = batch_label.to(device).long()
    if not opt.yz_celoss:
        logpy_z = encoder.log_prob(y.detach(), z.detach()) # 就是一个分类器
        encoder_d_loss = -logpy_z.mean()
    else:
        logit = encoder(z.detach())
        # encoder_d_loss = nn.CrossEntropyLoss()(logit, batch_label_softmax)
        encoder_d_loss = nn.CrossEntropyLoss()(logit, batch_label)
    encoder_d_loss.backward()
    encoder_opt.step()
    # Another forward since encoder was just updated
    for param in encoder.parameters():
        param.requires_grad = False
    if epoch > 0:
        encoder.eval()
        if not opt.yz_celoss:
            logpy_z = encoder.log_prob(y, z)    # Minimize py_z
            generator_d_loss = logpy_z.mean()
        else:
            logit = encoder(z)
            generator_d_loss = -nn.CrossEntropyLoss()(logit, batch_label_softmax)
    else:
        generator_d_loss = 0
    return generator_d_loss

def yx_disentangle_loss(encoder, encoder_opt, y, x):
    # use actual y
    encoder.train()
    encoder_opt.zero_grad()
    logpy_x = encoder.log_prob(y.detach(), x.detach())
    encoder_d_loss = -logpy_x.mean()
    encoder_d_loss.backward()
    encoder_opt.step()
    # Another forward since encoder was just updated
    encoder.eval()
    logpy_x = encoder.log_prob(y, x)
    generator_d_loss = -logpy_x.mean()  # Maximize py_x
    return generator_d_loss

def zx_disentangle_loss(encoder, encoder_opt, z, x):
    # use actual y
    encoder.train()
    encoder_opt.zero_grad()
    logpz_x = encoder.log_prob(z.detach(), x.detach())
    encoder_d_loss = -logpz_x.mean()
    encoder_d_loss.backward()
    encoder_opt.step()
    # Another forward since encoder was just updated
    encoder.eval()
    logpz_x = encoder.log_prob(z, x)
    generator_d_loss = -logpz_x.mean()  # Maximize pz_x
    return generator_d_loss
