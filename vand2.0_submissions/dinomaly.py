import random
import torch
from Dinomaly.dinamoly_model import dinomaly
from torch import nn
from types import SimpleNamespace
from util import cal_anomaly_maps_dino,  get_gaussian_kernel
import numpy as np
import torchvision.transforms as transforms
import colorful as cf

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


setup_seed(1)


class DinomalyModel(nn.Module):
    def __init__(self, class_name='carpet'):
        super(DinomalyModel, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.size = 256
        self.model = None
        self.normalize = transforms.Normalize(
            mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.CenterCrop(256),

            self.normalize,
        ])
        self.class_names = ['carpet', 'grid', 'leather', 'tile', 'wood',
                            'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
                            'pill', 'screw', 'toothbrush', 'transistor', 'zipper',]

        self.gaussian_kernel = get_gaussian_kernel(
            kernel_size=5, sigma=4).to('cuda')
        self.load_state_dict_()

    def load_state_dict_(self, model_path='weights/iter_3999_model.pth'):
        self.model = dinomaly()  # encoder_name='dinov2reg_vit_base_14')
        # load the model
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to(self.device)
        print(cf.bold & cf.blue | 'Dynomaly Model loaded successfully')

    def load_state_dict(self, state_dict):
        pass

    def train(self):
        self.model.train()

    @torch.no_grad()
    def __call__(self, imgs):
        """Transform the input batch and pass it through the model.

        This model returns a dictionary with the following keys:
        - ``anomaly_map`` - Anomaly map.
        - ``pred_score`` - Predicted anomaly score.
        """
        imgs = self.transform(imgs)
        feats, feats_pred = self.model(imgs)
        anomaly_map, _ = cal_anomaly_maps_dino(feats, feats_pred)
        # Only for dinomaly

        anomaly_map = self.gaussian_kernel(anomaly_map)
        anomaly_map = anomaly_map.squeeze().cpu().detach().numpy()
        pred_score = np.max(anomaly_map)
        # anomaly_map = anomaly_map.flatten(1)
        # pred_score = torch.sort(anomaly_map, dim=1, descending=True)[
        #     0][:, :int(anomaly_map.shape[1] * 0.01)]
        # pred_score = pred_score .mean(dim=1)
        pred_score = torch.tensor(pred_score)
        # resize the anomaly map to the original size
        anomaly_map = torch.from_numpy(anomaly_map)
        return {'pred_score': pred_score, 'anomaly_map': anomaly_map}

    def to(self, device):
        self.model.to(device)
