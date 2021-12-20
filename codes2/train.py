import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import os.path as osp
import time

from config import opt
from materials import image_feature, image_featureGZ
import util
from models import model
from models.resnet import make_resnet18_base
from lib import WeightedL1, calc_gradient_penalty, loss_fn, mse_loss, contrastive_loss, get_p_loss
from lib import yz_disentangle_loss, zx_disentangle_loss, yx_disentangle_loss, zy_disentangle_loss, unconstrained_z_loss, get_p_loss
from tcvae.tcvae import anneal_kl
from survae.distributions.conditional.normal import ConditionalNormal


# if __name__ == '__main__':
# gpu设置
global device
# device = torch.device('cpu')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.manualSeed)

# Dimension reduction on attributes,是否要对属性特征降维
if opt.pca_attribute != 0:
    opt.attSize = opt.pca_attribute

# load data
train_loader, test_loader, semanticfile = None, None, None
if opt.dataset == 'RAF':
    train_loader, test_loader = image_featureGZ.getRAFdata()
    semanticfile = 'materials/RAFsemantic'
elif opt.dataset == 'AffcetNet':
    train_loader, test_loader = image_feature.getAffectdata()
    semanticfile = 'materials/AFsemantic'
elif opt.dataset == 'FERPLUS':
    train_loader, test_loader = image_feature.getFERdata()
    semanticfile = 'materials/AFsemantic'
# resnet18提取图片特征
cnn = make_resnet18_base()
sd = cnn.state_dict()
sd.update(torch.load('materials/resnet18-base.pth'))
cnn.load_state_dict(sd)
cnn.to(device)

print('Current Training time', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

# 实验名称设定
exp_name = util.get_exp_name(opt)
print("Train Experiment: %s" % exp_name)

'''
if opt.survae:
    survae = SurVAEAlpha6(opt.resSize, opt.attSize, opt.decoder_layer_sizes[0], a1=opt.a1)
    netE = SurVAE_Encoder(survae)
    netG = SurVAE_Generator(survae)
else:
    netE = model.Encoder(opt)
    netG = model.Generator(opt)
'''
netE = model.Encoder(opt).to(device)   # 编码器
netG = model.Generator(opt).to(device)    # 生成器
netD = model.Discriminator_D1(opt).to(device)      # 辨别器
netF = model.Feedback(opt).to(device)
netDec = model.AttDec(opt, opt.attSize).to(device)
netS = None

###########
# Init Tensors
# input_res = torch.FloatTensor(opt.batch_size, opt.resSize).to(device)
# input_att = torch.FloatTensor(opt.batch_size, opt.attSize) .to(device) # attSize class-embedding size
# noise = torch.FloatTensor(opt.batch_size, opt.nz).to(device)
# one = torch.FloatTensor([1])
one = torch.tensor(1, dtype=torch.float).to(device)
mone = one * -1

# Optimizer
'''
if opt.survae:
    optimizer = optim.Adam(survae.parameters(), lr=1e-5, betas=(opt.beta1, 0.999))
    clip_grad_norm_(survae.parameters(), 20)
else:
    optimizer = optim.Adam(netE.parameters(), lr=opt.lr)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
'''
optimizer = optim.Adam(netE.parameters(), lr=opt.lr)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerF = optim.Adam(netF.parameters(), lr=opt.feed_lr, betas=(opt.beta1, 0.999))
optimizerDec = optim.Adam(netDec.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))

'''
# 这里本来是一段特征分类的预训练
# Prepare loss specific modules
if (opt.contrastive_loss and opt.contra_v != 3) or opt.z_loss:
    seen_clf = get_pretrain_classifier(data, opt)
else:
    seen_clf = None
'''
seen_clf = None

# 虽然整个p_loss没太看懂是做什么的，原代码的注解是Prototype loss，也就是原型的损失，但是论文里并没有提到这一点
# 从原代码里实现来看似乎是在计算同一个类别的所有特征的平均值
if opt.p_loss:
    prototypes = torch.zeros(train_loader.dataset.allclasses, opt.resSize).to(device)

if opt.zy_disentangle:
    net1 = nn.Sequential(nn.Linear(opt.attSize, 1000),
                         nn.ReLU(),
                         nn.Linear(1000, opt.latentSize*2))
    z_encoder = ConditionalNormal(net1).to(device)
    optimizerZ = optim.Adam(z_encoder.parameters(), lr=0.0001, betas=(opt.beta1, 0.99))

if opt.yz_disentangle:
    if not opt.yz_celoss:
        net2 = nn.Sequential(nn.Linear(opt.latentSize, 1000),
                             nn.ReLU(),
                             nn.Linear(1000, opt.attSize*2))
        yz_encoder = ConditionalNormal(net2, relu_mu=True).to(device)
        optimizerYZ = optim.Adam(yz_encoder.parameters(), lr=0.000001, betas=(opt.beta1, 0.999))
    else:
        yz_encoder = nn.Linear(opt.latentSize, train_loader.dataset.nseenclasses).to(device)
        optimizerYZ = optim.Adam(yz_encoder.parameters(), lr=0.00001, betas=(opt.beta1, 0.999))
if opt.yx_disentangle:
    net3 = nn.Sequential(nn.Linear(opt.resSize, 2000),
                         nn.ReLU(),
                         nn.Linear(2000, opt.attSize * 2))
    yx_encoder = ConditionalNormal(net3).to(device)
    optimizerYX = optim.Adam(yx_encoder.parameters(), lr=0.0000001, betas=(opt.beta1, 0.999))
if opt.zx_disentangle:
    net4 = nn.Sequential(nn.Linear(opt.resSize, 2000),
                         nn.ReLU(),
                         nn.Linear(2000, opt.latentSize * 2))
    zx_encoder = ConditionalNormal(net4).to(device)
    optimizerZX = optim.Adam(zx_encoder.parameters(), lr=0.00001, betas=(opt.beta1, 0.999))

# Prepare summary writer
if not opt.debug:
    log_dir = "runs/%s" % ('GZ'+exp_name)
    writer = SummaryWriter(log_dir)

best_gzsl_acc = 0
best_zsl_acc = 0
best_epoch = 0

for epoch in range(opt.continue_from, opt.nepoch):
    torch.cuda.empty_cache()
    netG.train()
    netDec.train()
    netF.train()
    netE.train()
    netD.train()
    cnn.train()
    D_cost_array = []
    G_cost_array = []
    WD_array = []
    vae_loss_array = []
    contrastive_loss_array = []
    zy_disentangle_loss_array = []
    yz_disentangle_loss_array = []
    zx_disentangle_loss_array = []
    yx_disentangle_loss_array = []
    unconstrained_z_loss_array = []
    p_loss_array = []
    siamese_loss_array = []
    assert netD.training
    assert netE.training
    assert netF.training
    assert netG.training
    assert netDec.training

    seenSemanticVectors = torch.load(semanticfile)['seenSemanticVectors']

    for loop in range(0, opt.feedback_loop):

        ######### Discriminator training ##############
        for p in netD.parameters():  # unfreeze discrimator
            p.requires_grad = True
        for p in netDec.parameters():  # unfreeze deocder
            p.requires_grad = True

        # Train D1 and Decoder (and Decoder Discriminator)
        gp_sum = 0  # lAMBDA VARIABLE
        for iter_d in range(opt.critic_iter):
            # 图片读取设置
            flag_not = 0
            train_size = 0
            pool_indices = []
            for i, (data, label, index) in enumerate(train_loader, 1):
                if data is None:
                    flag_not += 1
                    continue
                else:
                    train_size += data.size(0)
                    data = data.to(device)
                    input_l = label.to(device)
                    noise = torch.FloatTensor(data.size(0), opt.nz).to(device) # 随机生成一个noise
                    input_res, indices = cnn(data)
                    pool_indices.append(indices)
                    input_att = seenSemanticVectors[input_l].to(device) # 原文的属性特征（本文的动作特征）用语义特征初始化

                    netD.zero_grad()
                    input_resv = Variable(input_res)
                    input_attv = Variable(input_att)

                    netDec.zero_grad()
                    # step1, 直接从图片输入特征计算重构的特征，并将这个重构的特征和属性特征计算差异
                    recons = netDec(input_resv) # 一次fc后leakyrelu，一次fc后sigmoid(或另一种方式）激活，从下面的激活函数来看应该是在计算属性特征
                    if opt.attdec_use_mse:
                        R_cost = nn.MSELoss()(recons, input_attv)
                    else:
                        R_cost = WeightedL1(recons, input_attv)
                    R_cost.backward() # R_cost训练的是netDec
                    optimizerDec.step()

                    criticD_real = netD(input_resv, input_attv) # 把图片特征和属性特征链接，然后fc-leakyrelu-fc，输出的应该是一个向量
                    criticD_real = opt.gammaD * criticD_real.mean() # gammaD的解释是WGAN的参数，实际上没太明白这一步在做什么
                    criticD_real.requires_grad_(True)
                    criticD_real.backward(mone) # criticD_real训练的是netD

                    # 如果是encoded_noise的类型，就是用图片特征和属性特征做了一个分布，然后从这个分布里采样一个特征，否则直接从(0,1)分布采样
                    if opt.encoded_noise:
                        if not opt.survae:
                            means, log_var = netE(input_resv, input_attv)
                            std = torch.exp(0.5 * log_var)
                            eps = torch.randn([data.size(0), opt.latentSize]).cpu()
                            eps = Variable(eps.to(device))
                            z = eps * std + means  # torch.Size([64, 312])
                        else:
                            z, _ = netE(input_resv, input_attv)
                    else:
                        noise.normal_(0, 1)
                        z = Variable(noise)

                    if loop == 1:
                        fake = netG(z, c=input_attv) # 在上一步采样的特征+属性特征重构出来一个东西（我估计是样本特征+类别特征==>图片特征？）
                        dec_out = netDec(fake) # 再从这个生成的特征，计算属性特征
                        dec_hidden_feat = netDec.getLayersOutDet() # 在decoder中把没有经过sigmoid的特征返回来
                        feedback_out = netF(dec_hidden_feat) # fc-leakyRelu-fc-leakyRelu
                        fake = netG(z, a1=opt.a1, c=input_attv,
                                    feedback_layers=feedback_out) # 先做一遍，得到要feedback回来的值，然后再完整的得到类似残差连接的东西
                    else:
                        fake = netG(z, c=input_attv)

                    criticD_fake = netD(fake.detach(), input_attv) # 把生成的图片特征和属性特征链接，然后fc-leakyrelu-fc，输出的应该是一个向量
                    criticD_fake = opt.gammaD * criticD_fake.mean() # gammaD的解释是WGAN的参数，实际上没太明白这一步在做什么
                    criticD_fake.requires_grad_(True)
                    criticD_fake.backward(one) # 217-219是输入特征和输入属性的差别，这里是生成的特征和输入属性的车别
                    # gradient penalty 梯度惩罚策略
                    gradient_penalty = opt.gammaD * calc_gradient_penalty(device, netD, input_res, fake.data, input_att,
                                                                          opt)
                    # if opt.lambda_mult == 1.1:
                    gp_sum += gradient_penalty.data
                    gradient_penalty.backward()
                    Wasserstein_D = criticD_real - criticD_fake
                    WD_array.append(util.loss_to_float(Wasserstein_D))
                    D_cost = criticD_fake - criticD_real + gradient_penalty  # add Y here and #add vae reconstruction loss
                    D_cost_array.append(util.loss_to_float(D_cost))
                    optimizerD.step()
                gp_sum /= (opt.gammaD * opt.lambda1 * opt.critic_iter)
                if (gp_sum > 1.05).sum() > 0:
                    opt.lambda1 *= 1.1
                elif (gp_sum < 1.001).sum() > 0:
                    opt.lambda1 /= 1.1

                ############# Generator training ##############
                # Train Generator and Decoder
                for p in netD.parameters():  # freeze discrimator
                    p.requires_grad = False
                if opt.recons_weight > 0 and opt.freeze_dec:
                    for p in netDec.parameters():  # freeze decoder
                        p.requires_grad = False

                netE.zero_grad()
                netG.zero_grad()
                netF.zero_grad()
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)
                if not opt.survae:
                    means, log_var = netE(input_resv, input_attv)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([data.size(0), opt.latentSize]).cpu()
                    eps = Variable(eps.to(device))
                    z = eps * std + means  # torch.Size([64, 312])
                else:
                    z, _ = netE(input_resv, input_attv)
                if loop == 1:
                    recon_x = netG(z, c=input_attv)
                    dec_out = netDec(recon_x)
                    dec_hidden_feat = netDec.getLayersOutDet()
                    feedback_out = netF(dec_hidden_feat)
                    recon_x = netG(z, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
                else:
                    recon_x = netG(z, c=input_attv)
                    feedback_out = None

                beta = anneal_kl(opt, epoch)
                if not opt.survae:
                    input_eps = torch.randn([data.size(0), opt.resSize]).cpu().to(device)
                    hard_input = input_resv + opt.add_noise * input_eps
                    # 先计算重构的图片特征和添加扰动后的原始特征之间的距离(BCEloss)，然后计算一个KLloss，最后把两者相加
                    vae_loss_seen = loss_fn(recon_x, hard_input, means, log_var, opt, z,
                                            beta=beta)  # minimize E 3 with this setting feedback will update the loss as well
                else:
                    vae_loss_seen = netE.survae.vae_loss(input_resv, input_attv, feedback=feedback_out)
                    m_loss = mse_loss(input_resv, recon_x) * opt.m_lambda
                    if m_loss == m_loss:
                        vae_loss_seen += m_loss
                errG = vae_loss_seen
                vae_loss_array.append(util.loss_to_float(vae_loss_seen))

                # Contrastive loss，生成负样本对比损失
                if opt.contrastive_loss:
                    if loop == 1:
                        input_netF = netF
                    elif loop == 0:
                        input_netF = None
                    else:
                        assert False
                    contra_loss = contrastive_loss(device, netE, input_netF, netG, netDec, seen_clf,
                                                   input_resv, input_l, train_loader, seenSemanticVectors, opt,
                                                   deterministic=opt.train_deterministic,
                                                   temperature=opt.temperature, K=opt.K)
                    contrastive_loss_array.append(util.loss_to_float(contra_loss))
                    contra_loss *= opt.contra_lambda
                    contra_loss.backward(retain_graph=True)

                # p_loss
                if opt.p_loss:
                    p_loss = get_p_loss(netE, netF, netG, netDec, recon_x, input_attv, prototypes[input_l], opt)
                    p_loss_array.append(util.loss_to_float(p_loss))
                    p_loss *= opt.p_loss_lambda
                    p_loss.backward(retain_graph=True)

                # ZY disentangle loss
                if opt.zy_disentangle:
                    disentangle_loss = zy_disentangle_loss(z_encoder, optimizerZ, z, input_attv)
                    zy_disentangle_loss_array.append(util.loss_to_float(disentangle_loss))
                    disentangle_loss *= opt.zy_lambda
                    disentangle_loss.backward(retain_graph=True)
                # YZ disentangle loss
                if opt.yz_disentangle:
                    disentangle_loss = yz_disentangle_loss(device, yz_encoder, optimizerYZ, input_attv, z, epoch,
                                                           input_l, data, opt)
                    if disentangle_loss != disentangle_loss:
                        disentangle_loss = 0
                    yz_disentangle_loss_array.append(util.loss_to_float(disentangle_loss))
                    if disentangle_loss != 0:
                        disentangle_loss *= opt.yz_lambda
                        disentangle_loss.backward(retain_graph=True)
                # YX disentangle loss
                if opt.yx_disentangle:
                    disentangle_loss = yx_disentangle_loss(yx_encoder, optimizerYX, input_attv, recon_x)
                    if disentangle_loss != disentangle_loss:
                        disentangle_loss = 0
                    yx_disentangle_loss_array.append(util.loss_to_float(disentangle_loss))
                    if disentangle_loss != 0:
                        disentangle_loss *= opt.yx_lambda
                        disentangle_loss.backward(retain_graph=True)
                # ZX disentangle loss
                if opt.zx_disentangle:
                    disentangle_loss = zx_disentangle_loss(zx_encoder, optimizerZX, z, recon_x)
                    if disentangle_loss != disentangle_loss:
                        disentangle_loss = 0
                    zx_disentangle_loss_array.append(util.loss_to_float(disentangle_loss))
                    if disentangle_loss != 0:
                        disentangle_loss *= opt.zx_lambda
                        disentangle_loss.backward(retain_graph=True)

                # Unconstrained Z loss
                if opt.z_loss:
                    z_loss = unconstrained_z_loss(data, seen_clf, netG, netF, netDec, opt)
                    unconstrained_z_loss_array.append(util.loss_to_float(z_loss))
                    z_loss *= opt.z_loss_lambda
                    z_loss.backward(retain_graph=True)

                if opt.encoded_noise:
                    criticG_fake = netD(recon_x, input_attv).mean()
                    fake = recon_x
                else:
                    noise.normal_(0, 1)
                    noisev = Variable(noise)
                    if loop == 1:
                        fake = netG(noisev, c=input_attv)
                        dec_out = netDec(recon_x)  # Feedback from Decoder encoded output
                        dec_hidden_feat = netDec.getLayersOutDet()
                        feedback_out = netF(dec_hidden_feat)
                        fake = netG(noisev, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
                    else:
                        fake = netG(noisev, c=input_attv)
                    criticG_fake = netD(fake, input_attv).mean()

                G_cost = -criticG_fake
                errG += opt.gammaG * G_cost
                G_cost_array.append(util.loss_to_float(G_cost))
                netDec.zero_grad()
                recons_fake = netDec(fake)
                if opt.attdec_use_mse:
                    R_cost = nn.MSELoss()(recons_fake, input_attv)
                else:
                    R_cost = WeightedL1(recons_fake, input_attv)
                errG += opt.recons_weight * R_cost
                errG.backward()
                # write a condition here
                if opt.survae:
                    optimizer.step()
                else:
                    optimizer.step()
                    optimizerG.step()
                if loop == 1:
                    optimizerF.step()
                if opt.recons_weight > 0 and not opt.freeze_dec:  # not train decoder at feedback time
                    optimizerDec.step()
        # print('[%d/%d]  Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f' %
        #       (epoch, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(), vae_loss_seen.item()))
        print('[%d/%d]  Loss_D: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f' %
              (epoch+1, opt.nepoch, D_cost.item(), Wasserstein_D.item(), vae_loss_seen.item()))

        # Log
        D_cost_mean = np.array(D_cost_array).mean()
        G_cost_mean = np.array(G_cost_array).mean()
        WD_mean = np.array(WD_array).mean()
        vae_loss_mean = np.array(vae_loss_array).mean()
        if not opt.debug:
            writer.add_scalar("d_cost", D_cost_mean, epoch)
            writer.add_scalar("g_cost", D_cost_mean, epoch)
            writer.add_scalar("wasserstein_dist", WD_mean, epoch)
            writer.add_scalar("vae_loss", vae_loss_mean, epoch)
            if opt.contrastive_loss:
                contrastive_loss_mean = np.array(contrastive_loss_array).mean()
                writer.add_scalar("contrastive_loss", contrastive_loss_mean, epoch)
            if opt.zy_disentangle:
                disentangle_loss_mean = np.array(zy_disentangle_loss_array).mean()
                writer.add_scalar("zy_disentangle_loss", disentangle_loss_mean, epoch)
            if opt.yz_disentangle:
                disentangle_loss_mean = np.array(yz_disentangle_loss_array).mean()
                writer.add_scalar("yz_disentangle_loss", disentangle_loss_mean, epoch)
                print("YZ: %.3f" % disentangle_loss_mean)
            if opt.zx_disentangle:
                disentangle_loss_mean = np.array(zx_disentangle_loss_array).mean()
                writer.add_scalar("zx_disentangle_loss", disentangle_loss_mean, epoch)
                print("ZX: %.3f" % disentangle_loss_mean)
            if opt.yx_disentangle:
                disentangle_loss_mean = np.array(yx_disentangle_loss_array).mean()
                writer.add_scalar("yx_disentangle_loss", disentangle_loss_mean, epoch)
            if opt.z_loss:
                z_loss_mean = np.array(unconstrained_z_loss_array).mean()
                writer.add_scalar("z_loss", z_loss_mean)
            if opt.p_loss:
                p_loss_mean = np.array(p_loss_array).mean()
                writer.add_scalar("p_loss", p_loss_mean)

    state = {'netDec': netDec.state_dict(),
             'netF': netF.state_dict(),
             'netE': netE.state_dict(),
             'netD': netD.state_dict(),
             'netG': netG.state_dict(),
             'cnn': cnn.state_dict()}
    torch.save(state, osp.join('runs/'+'GZ'+exp_name, str(epoch+1)+'netState'))

    # test
    best_ave_acc = 0
    best_ave_acc_y = 0
    best_epoch_y = 0
    if (epoch+1)%opt.val_interval == 0:
        print('\n')
        netG.eval()
        netDec.eval()
        netF.eval()
        netE.eval()
        netD.eval()
        assert not netD.training
        assert not netE.training
        assert not netF.training
        assert not netG.training
        assert not netDec.training

        num_classes = 6

        classifier = nn.Linear(cnn.out_channels, num_classes)
        classifier.to(device)
        classifier.eval()

        classifier_y = nn.Linear(opt.attSize, num_classes)
        classifier_y.to(device)
        classifier_y.eval()

        loss_eval = nn.CrossEntropyLoss()
        loss_eval = loss_eval.to(device)
        ave_loss = None
        ave_acc = None
        ave_acc = []
        ave_acc_y = []
        ave_loss = []
        ave_loss_y = []

        eval_nodes = train_loader.dataset.train_nodes
        for label_i, label in enumerate(eval_nodes):
            torch.cuda.empty_cache()
            subloader = test_loader[label_i]
            acc = []
            acc_y = []
            # evaluate(device, subloader, eval_nodes[label_i])
            for i, (data, label, index) in enumerate(subloader, 1):
                if data is None:
                    continue
                else:
                    with torch.no_grad():
                        data = data.to(device)
                        label = label.to(device)
                        feature = cnn(data)
                        logits = classifier(feature)
                        feature = feature.to(device)
                        feature_y = netDec(feature)
                        feature_y = feature_y.to(device)
                        logits_y = classifier_y(feature_y)
                        ave_loss.append(util.loss_to_float(loss_eval(logits, label)))
                        ave_loss_y.append(util.loss_to_float(loss_eval(logits_y, label)))
                        _, pred = torch.max(logits, dim=1)
                        _, pred_y = torch.max(logits_y, dim=1)
                        tmp = torch.eq(pred, label).type(torch.FloatTensor).mean().item()
                        acc.append(tmp)
                        ave_acc.append(tmp)
                        tmp_y = torch.eq(pred_y, label).type(torch.FloatTensor).mean().item()
                        acc_y.append(tmp_y)
                        ave_acc_y.append(tmp_y)
            print("Average acc of origin input feature of %s is %.4f, and the acc of disentangled feature is %.4f"
                  % (eval_nodes[label_i], np.mean(acc), np.mean(acc_y)))
        writer.add_scalar("ave_acc", np.mean(ave_acc), epoch)
        writer.add_scalar("ave_acc_y", np.mean(ave_acc_y), epoch)
        writer.add_scalar("ave_loss", np.mean(ave_loss), epoch)
        writer.add_scalar("ave_loss_y", np.mean(ave_loss_y), epoch)
        print('Average acc of origin input feature of %s dataset is %.4f, and the acc of disentangled feature is %.4f'
              % (opt.dataset, np.mean(ave_acc), np.mean(ave_acc_y)))
        if np.mean(ave_acc) > best_ave_acc:
            best_ave_acc = np.mean(ave_acc)
        if np.mean(ave_acc_y) > best_ave_acc_y:
            best_ave_acc_y = np.mean(ave_acc_y)
            best_epoch_y = epoch+1
        print('\n')

print('Best acc of disentangled feature is %.4f, and is obtained at %s epoch'
      % (best_ave_acc_y, str(epoch+1)))

