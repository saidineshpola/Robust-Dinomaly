import torch
from torch import nn
from dinomaly import DinomalyModel
#from invad import ADerLoader
#from invad_lite import ADerLoader as ADerLoaderLite


class EnsembleModel(nn.Module):
    def __init__(self, class_name='carpet', device='cuda'):
        super(EnsembleModel, self).__init__()
        self.model1 = DinomalyModel()  # ADerLoaderLite()
        self.model2 = None  # DinomalyModel()
        self.device = device
        self.class_names = ['carpet', 'grid', 'leather', 'tile', 'wood',
                            'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
                            'pill', 'screw', 'toothbrush', 'transistor', 'zipper',]
        self.class_name = class_name
        self.use_second = False
        self.avg = False

    def forward(self, imgs):
        """
        Transform the input batch and pass it through the selected model.
        If self.use_second is True, use model2. Otherwise, use model1.
        """

        if self.use_second:
            output = self.model2(imgs)

        output = self.model1(imgs)

        pred_score = output['pred_score']
        anomaly_map = output['anomaly_map']

        return {'pred_score': pred_score, 'anomaly_map': anomaly_map}

    def load_state_dict(self, state_dict):
        pass

    def train(self):
        self.model1.train()
        self.model2.train()

    def to(self, device):
        self.model1.to(device)
        if self.model2 is not None:
            self.model2.to(device)
        self.device = device
