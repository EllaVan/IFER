import argparse
parser = argparse.ArgumentParser()

# gpu设置
parser.add_argument('--manualSeed', type=int, default=20, help='manual seed')
# parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')

# 数据集设置
parser.add_argument('--dataset', type=str, default='RAF', help='The used dataset')

#训练设置
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--concat_hy', type=int, default=1, help='Concat h and y during classification')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')

parser.add_argument('--additional', type=str, default="", help='additional str to add to exp name')
parser.add_argument('--val_interval', type=int, default=10, help='validation interval')
parser.add_argument('--save_interval', type=int, default=50, help='save interval')
parser.add_argument('--continue_from', type=int, default=0, help='save interval')

parser.add_argument('--lr', type=float, default=0.00001, help='learning rate to train GANs ')
parser.add_argument('--feed_lr', type=float, default=0.000001, help='learning rate to train GANs ')
parser.add_argument('--dec_lr', type=float, default=0.000001, help='learning rate to train GANs ')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lambda2', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')

parser.add_argument('--resSize', type=int, default=512, help='size of visual features')
parser.add_argument('--attSize', type=int, default=300, help='size of semantic features')
parser.add_argument('--nz', type=int, default=200, help='size of the latent z vector')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument("--latentSize", type=int, default=200)
parser.add_argument('--ngh', type=int, default=1024, help='size of the hidden units in generator')

parser.add_argument("--z_loss", action="store_true", default=False, help="Unconstrained z loss")
parser.add_argument("--z_loss_lambda", type=float, default=1.0, help="Scaling factor for unconstrained z loss")
parser.add_argument("--p_loss", action="store_true", default=False, help="Prototype loss")
parser.add_argument("--p_loss_lambda", type=float, default=1.0, help="Scaling factor for prototype loss")

parser.add_argument("--debug", action="store_true", default=False, help="Turn on debug mode")

parser.add_argument("--zy_disentangle", action="store_true", default=True, help="Use disentangle loss for y->z")
parser.add_argument("--zy_lambda", type=float, default=0.01, help="Scaling factor for zy disentangling loss")

parser.add_argument("--yz_disentangle", action="store_true", default=True, help="Use disentangle loss for z->y")
parser.add_argument("--yz_lambda", type=float, default=0.01, help="Scaling factor for yz disentangling loss")
parser.add_argument("--yz_celoss", action="store_true", default=False, help="Use cross entropy loss for z->l")

parser.add_argument("--yx_disentangle", action="store_true", default=True, help="Use disentangle loss for x->y")
parser.add_argument("--yx_lambda", type=float, default=0.01, help="Scaling factor for yx disentangling loss")

parser.add_argument("--zx_disentangle", action="store_true", default=True, help="Use disentangle loss for x->z")
parser.add_argument("--zx_lambda", type=float, default=0.01, help="Scaling factor for zx disentangling loss")

parser.add_argument("--z_disentangle", action="store_true", default=True, help="Use z disentangle loss")
parser.add_argument("--zd_beta", type=float, default=1.0, help="beta for scaling KL loss")
parser.add_argument("--zd_tcvae", action="store_true", default=False, help="Use TCVAE")
parser.add_argument("--zd_beta_annealing", action="store_true", default=False, help="Slowly increase beta")

parser.add_argument("--contrastive_loss", action="store_true", default=True, help="Use contrastive loss")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for contrastive loss")
parser.add_argument("--contra_lambda", type=float, default=1.0, help="Scaling factor of contrastive loss")
parser.add_argument("--contra_v", type=int, default=3, help="Version of contra loss to be used")
parser.add_argument("--K", type=int, default=5, help="Number of negative samples")

parser.add_argument("--siamese_loss", action="store_true", default=False, help="Train a Siamese network")
parser.add_argument("--siamese_lambda", type=float, default=1.0, help="Scaling factor for siamese network")
parser.add_argument("--siamese_use_softmax", action="store_true", default=False, help="Train a Siamese network")
parser.add_argument("--siamese_distance", type=str, default="l1", help="Distance metric for Siamese Net")

parser.add_argument("--pca_attribute", type=int, default=0, help="dimensionality reduction for attribute")

parser.add_argument('--feedback_loop', type=int, default=2) # 相当于增加残差链接
parser.add_argument('--encoded_noise', action='store_true', default=False, help='enables validation mode')

parser.add_argument('--survae', action='store_true', default=False, help='Use SurVAE model for encoder and generator')
parser.add_argument("--m_lambda", type=float, default=100.0, help="Strength for m_loss")

parser.add_argument("--add_noise", type=float, default=0.0, help="Add noise to reconstruction while training")
parser.add_argument("--recon", type=str, default="bce", help="VAE reconstruction loss: bce or l2 or l1")
parser.add_argument("--attdec_use_z", action="store_true", default=False, help="Use Z as additional input to attdec network")
parser.add_argument("--attdec_use_mse", action="store_true", default=False, help="Use MSE to calculate loss for attdec network")

parser.add_argument("--encoder_use_y", action="store_true", default=False, help="Encoder use y as input")
parser.add_argument("--train_deterministic", action="store_true", default=False, help="Deterministic sampling during training")
parser.add_argument('--gammaD', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--gammaG', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--gammaG_D2', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--gammaD2', type=int, default=1000, help='weight on the W-GAN loss')

parser.add_argument('--a1', type=float, default=1.0)
parser.add_argument('--a2', type=float, default=1.0)
parser.add_argument('--recons_weight', type=float, default=1.0, help='recons_weight for decoder')
parser.add_argument('--freeze_dec', action='store_true', default=False, help='Freeze Decoder for fake samples')

parser.add_argument("--encoder_layer_sizes", type=list, default=[512, 1024])
parser.add_argument("--decoder_layer_sizes", type=list, default=[1024, 512])

# 优化器设置
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

opt = parser.parse_args()