import torch
from sklearn import preprocessing
from sklearn.decomposition import PCA

def get_exp_name(opt):
    cf_option_str = [opt.dataset]

    # Encoder use Y as input
    if opt.encoder_use_y:
        addtional_str = 'encyx'
    else:
        addtional_str = 'encx'
    cf_option_str.append(addtional_str)

    # Train deterministic
    if opt.train_deterministic:
        addtional_str = 'deter'
        cf_option_str.append(addtional_str)

    # Use zy disentangle
    if opt.zy_disentangle:
        additional_str = "zydscale%.3f" % (opt.zy_lambda)
        cf_option_str.append(additional_str)

    # Use yz disentangle
    if opt.yz_disentangle:
        additional_str = "yzdscale%.3f" % (opt.yz_lambda)
        if opt.yz_celoss:
            additional_str += "cel"
        cf_option_str.append(additional_str)

    # Use yx disentangle
    if opt.yx_disentangle:
        additional_str = "yxdscale%.3f" % (opt.yx_lambda)
        cf_option_str.append(additional_str)

    # Use zx disentangle
    if opt.zx_disentangle:
        additional_str = "zxdscale%.3f" % (opt.zx_lambda)
        cf_option_str.append(additional_str)

    # Use z disentangle
    if opt.z_disentangle:
        additional_str = "zdbeta%.1ftcvae%danneal%d" % (opt.zd_beta, opt.zd_tcvae, opt.zd_beta_annealing)
        cf_option_str.append(additional_str)

    # Use contrastive loss
    if opt.contrastive_loss:
        additional_str = "contrav%dscale%.1ft%.1f" % (opt.contra_v, opt.contra_lambda, opt.temperature)
        if opt.K != 30:
            additional_str += "K%d" % opt.K
        cf_option_str.append(additional_str)

    # No feedback loop
    if opt.feedback_loop == 1:
        additional_str = "nofeedback"
        cf_option_str.append(additional_str)

    # Encoded noise
    if not opt.encoded_noise:
        additional_str = "noise"
        cf_option_str.append(additional_str)

    # Siamese loss
    if opt.siamese_loss:
        additional_str = "siamese%.1fsoftmax%ddist%s" % (opt.siamese_lambda, opt.siamese_use_softmax, opt.siamese_distance)
        cf_option_str.append(additional_str)

    # Latent size
    if opt.latentSize != 312:
        additional_str = "latent%d" % (opt.latentSize)
        cf_option_str.append(additional_str)

    # Attr PCA
    if opt.pca_attribute != 0:
        additional_str = "pca%d" % (opt.pca_attribute)
        cf_option_str.append(additional_str)

    # SurVAE
    if opt.survae:
        additional_str = "survae%.1f" % opt.m_lambda
        cf_option_str.append(additional_str)

    # Add noise
    if opt.add_noise != 0.0:
        additional_str = "noise%.2f" % opt.add_noise
        cf_option_str.append(additional_str)

    # VAE Reconstruction loss
    if opt.recon != "bce":
        additional_str = "recon%s" % (opt.recon)
        cf_option_str.append(additional_str)

    # Att Dec use z
    if opt.attdec_use_z:
        additional_str = "attdecz"
        cf_option_str.append(additional_str)

    # Att Dec use MSE loss
    if opt.attdec_use_mse:
        additional_str = "attdecmse"
        cf_option_str.append(additional_str)

    # Unconstrained Z loss
    if opt.z_loss:
        additional_str = "zloss%.1f" % opt.z_loss_lambda
        cf_option_str.append(additional_str)

    # Prototype loss
    if opt.p_loss:
        additional_str = "ploss%.3f" % opt.p_loss_lambda
        cf_option_str.append(additional_str)

    # Additional str
    if opt.additional != "":
        cf_option_str.append(opt.additional)
    return "-".join(cf_option_str)

def loss_to_float(loss):
    if isinstance(loss, torch.Tensor):
        return loss.item()
    else:
        return float(loss)
