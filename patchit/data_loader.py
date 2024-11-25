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
        
        :param data_dir: Path to the dataset directory.
        :param transform: Transformations to apply to the dataset.
        :param batch_size: Batch size for the DataLoader.
        :param custom_classes: Custom list of class names in desired order (optional).
        """

        # Define default transformations if none are provided
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.batch_size = batch_size
        self.data_dir = data_dir
        
        # Load dataset
        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)

        self.class_to_idx = self.dataset.class_to_idx
        
        # Create idx_to_class mapping
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Create DataLoader
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def get_sample_batch(self):
        """
        Get a single batch of data from the DataLoader.
        
        :return: Tuple (images, labels)
        """
        images, labels = next(iter(self.dataloader))

        self.images = images.clone()

        self.labels = labels.clone()

        return self.images, self.labels

    def plot_sample_batch(self):
        """
        Plot a batch of images with their corresponding labels.
        """
        images, labels = self.images, self.labels
        print(labels)
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

