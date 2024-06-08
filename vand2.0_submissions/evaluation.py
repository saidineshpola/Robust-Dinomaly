import argparse
import importlib
from pathlib import Path

import torch
from torch import nn

from anomalib.data import MVTec
from anomalib.metrics import F1Max
import colorful as cf
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.v2 import Compose, RandomAdjustSharpness, RandomHorizontalFlip, \
    Resize, ToTensor, RandomVerticalFlip, RandomRotation, ColorJitter, GaussianBlur


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_path", type=str, required=True)
    parser.add_argument("--class_name", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    return parser.parse_args()


def load_model(module_path: str, class_name: str, weights_path: str, category: str) -> nn.Module:
    """Load model.

    Args:
        module_path (str): Path to the module containing the model class.
        class_name (str): Name of the model class.
        weights_path (str): Path to the model weights.
        category (str): Category of the dataset.

    Note:
        We assume that the weight path contain the weights for all categories.
            For example, if the weight path is "/path/to/weights/", then the
            weights for each category should be stored as
            "/path/to/weights/bottle.pth", "/path/to/weights/zipper.pth", etc.

    Returns:
        nn.Module: Loaded model.
    """
    # get model class
    model_class = getattr(importlib.import_module(module_path), class_name)
    # instantiate model
    model = model_class(category)
    # load weights
    if weights_path:
        weight_file = Path(weights_path) / f"model.pth"
        model.load_state_dict(torch.load(weight_file))
    return model


def apply_transformations(image, mask):
    """Apply transformations to the image and mask.

    Args:
        image (PIL.Image or torch.Tensor): Input image.
        mask (PIL.Image or torch.Tensor): Corresponding mask.

    Returns:
        torch.Tensor, torch.Tensor: Transformed image and mask.
    """
    spatial_transforms = transforms.Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(20),
    ])

    image_transform = transforms.Compose([
        spatial_transforms,
        ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        GaussianBlur(3, sigma=(0.1, 2.0)),  # Apply Gaussian blur
        ToTensor(),
    ])

    mask_transform = transforms.Compose([
        spatial_transforms,
        ToTensor(),
    ])

    # Apply the same spatial transformations to both image and mask
    seed = torch.initial_seed()
    torch.manual_seed(seed)
    image = image_transform(image)

    torch.manual_seed(seed)
    mask = mask_transform(mask)

    return image, mask


def run(module_path: str, class_name: str, weights_path: str, dataset_path: str, category: str) -> None:
    """Run the evaluation script.

    Args:
        module_path (str): Path to the module containing the model class.
        class_name (str): Name of the model class.
        weights_path (str | None, optional): Path to the model weights.
        dataset_path (str): Path to the dataset.
        category (str): Category of the dataset.
    """
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Instantiate and load the model
    model = load_model(module_path, class_name, weights_path, category)
    model.to(device)

    # Create the dataset
    datamodule = MVTec(root=dataset_path, eval_batch_size=1,
                       image_size=[256, 256],
                       category=category)
    datamodule.setup()

    # Create the metrics
    image_metric = F1Max()
    pixel_metric = F1Max()

    # Loop over the test set and compute the metrics
    for data in tqdm(datamodule.test_dataloader()):
        # Apply transformations outside the dataset
        transformed_image, transformed_mask = apply_transformations(
            data["image"], data["mask"])
        # transformed_image, transformed_mask = (
        #     data["image"], data["mask"])
        output = model(transformed_image.to(device))

        # Update the image metric
        image_metric.update(output["pred_score"].cpu(), data["label"])

        # Update the pixel metric
        pixel_metric.update(
            output["anomaly_map"].squeeze().cpu(), transformed_mask.squeeze().cpu())

    # Compute the metrics
    image_score = image_metric.compute()
    pixel_score = pixel_metric.compute()
    print(cf.red | f'Category {category}')
    print(cf.green | f'Image F1 Max score {image_score}')
    print(cf.green | f'Pixel F1 Max score {pixel_score}')


if __name__ == "__main__":
    args = parse_args()
    run(args.module_path, args.class_name,
        args.weights_path, args.dataset_path, args.category)
