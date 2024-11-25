import matplotlib.pyplot as plt
import numpy as np
import torch

class Visualizer:
    def __init__(self, idx_to_class, adv_patch, model, norm_mean=None, norm_std=None):
        """
        Initialize the Visualizer.
        
        :param idx_to_class: Dictionary mapping indices to class labels.
        :param adv_patch: Adversarial patch object.
        :param model: Pre-trained model.
        :param norm_mean: Normalization mean used during preprocessing.
        :param norm_std: Normalization std used during preprocessing.
        """
        self.idx_to_class = idx_to_class
        self.norm_mean = np.array(norm_mean if norm_mean else [0.485, 0.456, 0.406])
        self.norm_std = np.array(norm_std if norm_std else [0.229, 0.224, 0.225])
        self.adv_patch = adv_patch
        self.device = adv_patch.device
        self.model = model

    def _unnormalize_image(self, img):
        """
        Unnormalize an image tensor.
        
        :param img: The image tensor (C, H, W).
        :return: Unnormalized numpy array (H, W, C).
        """
        img = img.cpu().permute(1, 2, 0).numpy()  # CHW -> HWC
        img = img * self.norm_std[None, None, :] + self.norm_mean[None, None, :]
        img = np.clip(img, 0, 1)  # Clip values to valid range
        return img

    def show_predictions(self, images, labels, patch, n_samples=5, top_k=5):
        """
        Visualize original images, patched images, and their predictions.

        :param images: Original image tensors (batch of C, H, W).
        :param labels: True labels for the images.
        :param patch: Adversarial patch tensor.
        :param n_samples: Number of samples to visualize (default: 5).
        :param top_k: Number of top predictions to display (default: 5).
        """
        # Generate predictions for original and patched images
        norm_images = images.clone()
        predictions = self.model(images.to(self.adv_patch.device))
        patched_images = self.adv_patch.place_patch(norm_images, patch)
        patched_predictions = self.model(patched_images.to(self.adv_patch.device))

        # Set up the visualization layout
        fig, axes = plt.subplots(
            n_samples, 4, figsize=(16, n_samples * 4)
        )  # 4 columns: Original, Original Predictions, Patched, Patched Predictions
        axes = np.atleast_2d(axes)  # Ensure a consistent 2D array for axes

        for i in range(n_samples):
            # Unnormalize the original image

            img = self._unnormalize_image(images[i])
            label = labels[i] if isinstance(labels[i], str) else labels[i].item()

            # Unnormalize the patched image
            patched_img = self._unnormalize_image(patched_images[i])

            # Original Image
            ax = axes[i, 0]
            ax.imshow(img)
            ax.set_title(f"Original: {self.idx_to_class[label] if isinstance(label, int) else label}")
            ax.axis("off")

            # Predictions for Original Image
            pred = predictions[i]
            if abs(pred.sum().item() - 1.0) > 1e-4:
                pred = torch.softmax(pred, dim=-1)
            topk_vals_orig, topk_idx_orig = pred.topk(top_k, dim=-1)
            topk_vals_orig, topk_idx_orig = topk_vals_orig.cpu().detach().numpy(), topk_idx_orig.cpu().detach().numpy()

            ax = axes[i, 1]
            ax.barh(
                np.arange(top_k),
                topk_vals_orig * 100.0,
                align="center",
                color=["C0" if topk_idx_orig[j] != label else "C2" for j in range(top_k)],
            )
            ax.set_yticks(np.arange(top_k))
            ax.set_yticklabels([self.idx_to_class[c] for c in topk_idx_orig])
            ax.invert_yaxis()
            ax.set_xlabel("Confidence (%)")
            ax.set_title("Original Predictions")

            # Patched Image
            ax = axes[i, 2]
            ax.imshow(patched_img)
            ax.set_title("Patched")
            ax.axis("off")

            # Predictions for Patched Image
            patched_pred = patched_predictions[i]
            if abs(patched_pred.sum().item() - 1.0) > 1e-4:
                patched_pred = torch.softmax(patched_pred, dim=-1)
            topk_vals_patch, topk_idx_patch = patched_pred.topk(top_k, dim=-1)
            topk_vals_patch, topk_idx_patch = topk_vals_patch.cpu().detach().numpy(), topk_idx_patch.cpu().detach().numpy()

            ax = axes[i, 3]
            ax.barh(
                np.arange(top_k),
                topk_vals_patch * 100.0,
                align="center",
                color=["C1" if topk_idx_patch[j] != label else "C2" for j in range(top_k)],
            )
            ax.set_yticks(np.arange(top_k))
            ax.set_yticklabels([self.idx_to_class[c] for c in topk_idx_patch])
            ax.invert_yaxis()
            ax.set_xlabel("Confidence (%)")
            ax.set_title("Patched Predictions")

        plt.tight_layout()
        plt.show()
