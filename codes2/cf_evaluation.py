import torch
from torch.autograd import Variable


class Evaluate():
    def __init__(self, device, netE, netG, netDec, netF, data, opt, model_file,
                 exp_name, clf_epoch, alpha=1.0, siamese=False, netS=None):
        netG.eval()
        netDec.eval()
        netF.eval()
        netE.eval()
        self.device = device
        self.netE = netE.to(device)
        self.netG = netG.to(device)
        self.netDec = netDec.to(device)
        if opt.feedback_loop == 1:
            self.netF = None
        else:
            self.netF = netF.to(device)
        self.data = data
        self.opt = opt
        self.model_file = model_file
        self.exp_name = exp_name
        self.epoch = clf_epoch
        if opt.concat_hy:
            self.cls_netDec = self.netDec
        else:
            self.cls_netDec = None
        self.alpha = alpha
        self.siamese = siamese

    def conditional_sample(self, x, y, deterministic=False):
        with torch.no_grad():
            if not self.opt.survae:
                means, log_var = self.netE(x, y)
                if deterministic:
                    z = means
                else:
                    z = torch.normal(means, torch.exp(0.5*log_var))
            else:
                z, _ = self.netE(x, y)
            zv = Variable(z)
            yv = Variable(y)
            x_gen = self.netG(zv, c=yv)
            if self.netF is not None:
                _ = self.netDec(x_gen)
                dec_hidden_feat = self.netDec.getLayersOutDet()  # no detach layers
                feedback_out = self.netF(dec_hidden_feat)
                x_gen = self.netG(zv, a1=self.opt.a2, c=yv, feedback_layers=feedback_out)
        return x_gen

    def generate_syn_feature_cf(self, x, classes, deterministic=False):
        attribute = self.data.attribute
        nclass = classes.size(0)
        opt = self.opt
        device = self.device
        num = opt.syn_num
        syn_feature = torch.zeros(nclass * num, opt.resSize).to(device)
        syn_label = torch.zeros(nclass * num).long().to(device)
        syn_att = torch.zeros(num, opt.attSize).float().to(device)
        syn_noise = torch.zeros(num, opt.nz).float().to(device)
        with torch.no_grad():
            for i in range(nclass):
                iclass = classes[i]
                iclass_att = attribute[iclass]
                if not self.opt.survae:
                    means, log_var = self.netE(x.unsqueeze(0), iclass_att.unsqueeze(0))
                    means = means.expand(num, -1)
                    log_var = log_var.expand(num, -1)
                    syn_att.copy_(iclass_att.repeat(num, 1))
                    if deterministic:
                        syn_noise = means
                    else:
                        syn_noise = torch.normal(means, torch.exp(0.5 * log_var))
                else:
                    syn_noise, _ = self.netE(x.unsqueeze(0), iclass_att.unsqueeze(0))
                    syn_noise = syn_noise.expand(num, -1)
                syn_noisev = Variable(syn_noise)
                syn_attv = Variable(syn_att)
                fake = self.netG(syn_noisev, c=syn_attv)
                if self.netF is not None:
                    dec_out = self.netDec(fake)  # only to call the forward function of decoder
                    dec_hidden_feat = self.netDec.getLayersOutDet()  # no detach layers
                    feedback_out = self.netF(dec_hidden_feat)
                    fake = self.netG(syn_noisev, a1=opt.a2, c=syn_attv, feedback_layers=feedback_out)
                output = fake
                syn_feature.narrow(0, i * num, num).copy_(output.data)
                syn_label.narrow(0, i * num, num).fill_(iclass)
        return syn_feature, syn_label

    def zsl(self, softmax_clf, cf, deterministic=False):
        opt = self.opt
        data = self.data
        device = self.device
        if not cf:
            with torch.not_grad():
                gen_x, gen_l = self.generate_syn_feature(self.netG, self.data.unseenclasses, self.data.attribute,
                                                    opt.syn_num, netF=self.netF, netDec=self.netDec, opt=opt)
            if softmax_clf:
                pass


