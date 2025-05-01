import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from src.ML_TEMPERATURE_PREDICTION.entity.config_entity import DataTransformationConfig
from src.ML_TEMPERATURE_PREDICTION.logging import logger
import tifffile

class RGBThermalDataset(Dataset):
    """
    Dataset for paired RGB and thermal images
    """
    def __init__(self, rgb_dir, thermal_dir, transform=None, mode='train'):
        self.rgb_dir = rgb_dir
        self.thermal_dir = thermal_dir
        self.transform = transform
        self.mode = mode
        
        # Get list of image filenames (without extensions)
        self.rgb_files = [f.split('.')[0] for f in os.listdir(rgb_dir) if f.endswith('.jpg')]
        self.thermal_files = [f.split('.')[0] for f in os.listdir(thermal_dir) if f.endswith('.tiff')]
        
        # Get common filenames (images that have both RGB and thermal versions)
        self.files = list(set(self.rgb_files).intersection(set(self.thermal_files)))
        logger.info(f"Found {len(self.files)} paired images in {mode} set")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        filename = self.files[index]
        
        # Load RGB image
        rgb_path = os.path.join(self.rgb_dir, f"{filename}.jpg")
        rgb_img = Image.open(rgb_path).convert('RGB')
        
        # Load thermal image (16-bit TIFF)
        thermal_path = os.path.join(self.thermal_dir, f"{filename}.tiff")
        thermal_img = tifffile.imread(thermal_path)
        
        # Normalize thermal image to 0-1 range
        thermal_img = thermal_img.astype(np.float32)
        thermal_min = np.min(thermal_img)
        thermal_max = np.max(thermal_img)
        if thermal_max > thermal_min:
            thermal_img = (thermal_img - thermal_min) / (thermal_max - thermal_min)
        else:
            thermal_img = np.zeros_like(thermal_img)
        
        # Convert to PIL image for transforms
        thermal_img = Image.fromarray((thermal_img * 255).astype(np.uint8))
        
        # Apply transforms if provided
        if self.transform:
            rgb_img = self.transform(rgb_img)
            thermal_img = self.transform(thermal_img)
        
        return {'A': rgb_img, 'B': thermal_img, 'filename': filename}

class DataTransformation:
    """
    Data transformation component for preparing RGB and thermal image datasets
    """
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        
    def get_data_transforms(self):
        """
        Define the transformations to be applied to the images
        """
        # For Pix2Pix, images are resized and then random crops are taken
        transform = transforms.Compose([
            transforms.Resize(self.config.image_size + 30),
            transforms.RandomCrop(self.config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if self.config.channels == 3 else transforms.Normalize([0.5], [0.5])
        ])
        
        # For validation/test, just resize to target size (no random operations)
        transform_val = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.CenterCrop(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if self.config.channels == 3 else transforms.Normalize([0.5], [0.5])
        ])
        
        return transform, transform_val
    
    def get_data_loaders(self):
        """
        Create and return data loaders for training, validation, and testing
        """
        transform, transform_val = self.get_data_transforms()
        
        # Create datasets
        train_dataset = RGBThermalDataset(
            rgb_dir=os.path.join(self.config.data_path, 'train', 'rgb'),
            thermal_dir=os.path.join(self.config.data_path, 'train', 'thermal'),
            transform=transform,
            mode='train'
        )
        
        val_dataset = RGBThermalDataset(
            rgb_dir=os.path.join(self.config.data_path, 'val', 'rgb'),
            thermal_dir=os.path.join(self.config.data_path, 'val', 'thermal'),
            transform=transform_val,
            mode='val'
        )
        
        test_dataset = RGBThermalDataset(
            rgb_dir=os.path.join(self.config.data_path, 'test', 'rgb'),
            thermal_dir=os.path.join(self.config.data_path, 'test', 'thermal'),
            transform=transform_val,
            mode='test'
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        return train_loader, val_loader, test_loader