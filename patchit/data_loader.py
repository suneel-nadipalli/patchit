import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class DatasetLoader:
    def __init__(self, data_dir, transform=None, batch_size=16):
        """
        Initialize the DatasetLoader.
        
        :param data_dir: Path to the dataset root directory.
        :param transform: Transformations to apply to the images (default: standard transforms).
        :param batch_size: Number of samples per batch (default: 16).
        """
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Define default transformations if none are provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform

        # Load the dataset
        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Class-to-Index Mapping
        self.class_to_idx = self.dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def get_sample_batch(self):
        """
        Get a single batch of data from the DataLoader.
        
        :return: Tuple (images, labels)
        """
        images, labels = next(iter(self.dataloader))

        self.images = images.clone()

        self.labels = labels.clone()

        return images.clone(), labels.clone()

    def plot_sample_batch(self):
        """
        Plot a batch of images with their corresponding labels.
        """
        images, labels = self.get_sample_batch()
        fig, axes = plt.subplots(1, min(len(images), 5), figsize=(15, 5))  # Plot up to 5 images
        for i, ax in enumerate(axes):
            img = images[i].permute(1, 2, 0).numpy()  # Convert CHW to HWC
            # Unnormalize the image by converting the mean and std to numpy
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean  # Unnormalize
            img = img.clip(0, 1)  # Clip to valid range

            ax.imshow(img)
            ax.set_title(f"Class: {self.idx_to_class[labels[i].item()]}")
            ax.axis("off")
        plt.show()

