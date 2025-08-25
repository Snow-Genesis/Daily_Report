import os
import kagglehub

import torch
import torchvision.transforms as T

from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator


def DownloadDataset(save_dir: str) -> None:
    # Set path for downloading
    os.environ['KAGGLEHUB_CACHE'] = save_dir

    # Download the dataset from kaggle
    path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")
    print("Path to dataset files:", path)

class CustomizedDataset(ClassificationDataset):
    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        super().__init__(root, args, augment, prefix)

        # Transform for training dataset
        train_transforms = T.Compose(
            [
                T.Resize((args.imgsz, args.imgsz)),
                T.RandomHorizontalFlip(p=args.fliplr),
                T.RandomVerticalFlip(p=args.flipud),
                T.RandAugment(interpolation=T.InterpolationMode.BILINEAR),
                T.ColorJitter(brightness=args.hsv_v, contrast=args.hsv_v, saturation=args.hsv_s, hue=args.hsv_h),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
                T.RandomErasing(p=args.erasing, inplace=True),
            ]
        )

        # Transforms for validation dataset
        val_transforms = T.Compose(
            [
                T.Resize((args.imgsz, args.imgsz)),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
            ]
        )
        self.torch_transforms = train_transforms if augment else val_transforms


class CustomizedTrainer(ClassificationTrainer):
    """A customized trainer class for YOLO classification models with enhanced dataset handling."""

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """Build a customized dataset for classification training and the validation during training."""
        return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

class CustomizedValidator(ClassificationValidator):
    def build_dataset(self, img_path: str, mode: str='train'):
        return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=self.args.split)

def main() -> None:
    model = YOLO('yolo11n-cls.pt')
    model.train(data='./dataset/train/', trainer=ClassificationTrainer, epochs=10, imgsz=224, batch=10)
    model.val(data='./dataset/test/', validator=CustomizedValidator, imgsz=224, batch=10)

if __name__=='__main__':
    main()