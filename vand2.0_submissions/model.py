"""Sample Model class for track 1."""

import torch
from torch import nn
from torchvision.transforms import v2

# We use PatchcoreModel for an example. You can replace it with your model.
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


class Patchcore(nn.Module):
    """Examplary Model class for track 1.

    This class contains the torch model and the transformation pipeline.
    Forward-pass should first transform the input batch and then pass it through the model.

    Note:
        This is the example model class name. You can replace it with your model class name.

    Args:
        backbone (str): Name of the backbone model to use.
            Default: "wide_resnet50_2".
        layers (list[str]): List of layer names to use.
            Default: ["layer1", "layer2", "layer3"].
        pre_trained (bool): If True, use pre-trained weights.
            Default: True.
        num_neighbors (int): Number of neighbors to use.
            Default: 9.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: list[str] = ["layer1", "layer2", "layer3"],  # noqa: B006
        pre_trained: bool = True,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()

        # NOTE: Create your transformation pipeline here.
        # We use Resize, CenterCrop, and Normalize for an example.
        self.transform = v2.Compose(
            [
                v2.Resize((256, 256)),
                v2.CenterCrop((224, 224)),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False),
                # ... and some more transformations
            ],
        )

        # NOTE: Create your model here. We use PatchcoreModel for an example.
        self.model = PatchcoreModel(
            backbone=backbone,
            layers=layers,
            pre_trained=pre_trained,
            num_neighbors=num_neighbors,
        )

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Transform the input batch and pass it through the model.

        This model returns a dictionary with the following keys
        - ``anomaly_map`` - Anomaly map.
        - ``pred_score`` - Predicted anomaly score.
        """
        batch = self.transform(batch)
        return self.model(batch)
