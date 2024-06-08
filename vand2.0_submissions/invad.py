import torch
from ADer.model.invad import InvAD, invad
from ADer.model.dinamoly import dinamoly
from torch import nn
from types import SimpleNamespace
from util import cal_anomaly_maps
import numpy as np
import torchvision.transforms as transforms
import colorful as cf
torch.manual_seed(42)
np.random.seed(42)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ADerLoader(nn.Module):
    def __init__(self, reso=256):
        super().__init__()  # Add this line
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.size = reso
        self.model = None
        self.normalize = transforms.Normalize(
            mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.init_model_invad()
        self.class_names = ['carpet', 'grid', 'leather', 'tile', 'wood',
                            'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
                            'pill', 'screw', 'toothbrush', 'transistor', 'zipper',]
        self.class_routes = {
            'bottle': 'invad',
            'cable': 'invad',
            'capsule': 'invad',
            'carpet': 'invad',
            'grid': 'invad',
            'hazelnut': 'invad',
            'leather': 'invad',
            'metal_nut': 'invad',
            'pill': 'invad',
            'screw': 'invad',
            'tile': 'invad',
            'toothbrush': 'invad',
            'transistor': 'invad',
            'wood': 'invad',
            'zipper': 'invad',

        }

    def init_model_invad(self, model_type='default'):
        in_chas = [256, 512, 1024]
        out_cha = 64
        style_chas = [min(in_cha, out_cha) for in_cha in in_chas]
        in_strides = [2 ** (len(in_chas) - i - 1) for i in range(len(in_chas))]
        latent_channel_size = 16
        latent_spatial_size = self.size // (2 ** (1 + len(in_chas)))

        model_encoder = SimpleNamespace()
        model_encoder.name = 'timm_wide_resnet50_2'
        model_encoder.kwargs = dict(pretrained=False, checkpoint_path='ADer/model/pretrain/wide_resnet50_racm-8234f177.pth',
                                    strict=False, features_only=True, out_indices=[1, 2, 3])

        model_fuser = dict(
            type='Fuser',
            in_chas=in_chas,
            style_chas=style_chas,
            in_strides=[4, 2, 1],
            down_conv=True,
            bottle_num=1,
            conv_num=1,
            conv_type='conv',
            lr_mul=0.01
        )

        model_decoder = dict(
            in_chas=in_chas,
            style_chas=style_chas,
            latent_spatial_size=latent_spatial_size,
            latent_channel_size=latent_channel_size,
            blur_kernel=[1, 3, 3, 1],
            normalize_mode='LayerNorm',
            lr_mul=0.01,
            small_generator=True,
            layers=[2] * len(in_chas)
        )

        self.model = invad(
            pretrained=False,
            model_encoder=model_encoder,
            model_fuser=model_fuser,
            model_decoder=model_decoder,
        )
        self.load_state_dict()

    def load_state_dict(self, model_path='/home/saidinesh/Desktop/Projects/vand2.0_cvpr/ADer/runs/InvADTrainer_configs_invad_invad_mvtec_20240601-162705/net_10.pth'):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to(self.device)
        print(cf.bold & cf.blue | 'Ader Model loaded successfully')

    def train(self):
        self.model.train()

    def __call__(self, imgs):
        """Transform the input batch and pass it through the model.

        This model returns a dictionary with the following keys:
        - ``anomaly_map`` - Anomaly map.
        - ``pred_score`` - Predicted anomaly score.
        """
        imgs = self.normalize(imgs)
        feats, feats_pred = self.model(imgs)
        anomaly_map, _ = cal_anomaly_maps(feats, feats_pred)
        pred_score = np.max(anomaly_map)
        pred_score = torch.tensor(pred_score)
        anomaly_map = torch.from_numpy(anomaly_map)

        return {'pred_score': pred_score, 'anomaly_map': anomaly_map}

    def forward_d(self, imgs, detach=False):
        imgs = imgs.to(self.device)
        return self.model.forward_d(imgs, detach)

    def to(self, device):
        self.model.to(device)
