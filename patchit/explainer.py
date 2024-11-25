import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import Saliency, LayerGradCam

class Explainer:
    def __init__(self, model, device, adv_patch, norm_mean=None, norm_std=None):
        self.model = model
        self.device = device
        self.adv_patch = adv_patch
        self.norm_mean = np.array(norm_mean if norm_mean else [0.485, 0.456, 0.406])
        self.norm_std = np.array(norm_std if norm_std else [0.229, 0.224, 0.225])
        self.saliency = Saliency(self.model)

        self.get_last_conv_layer()

    def get_last_conv_layer(self):
        """
        Find the last convolutional layer in a given model.

        :param model: PyTorch model instance.
        :return: The last convolutional layer of the model.
        """
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2D layer found in the model.")
        
        self.layer_gc = LayerGradCam(self.model, last_conv) 

    
    def _unnormalize_image(self, img):
        """
        Unnormalize an image tensor for visualization.
        """
        img = img.cpu().permute(1, 2, 0).numpy()
        img = img * self.norm_std[None, None, :] + self.norm_mean[None, None, :]
        img = np.clip(img, 0, 1)
        return img

    def _plot_grad_cam(self, input_tensor, label):
        gradcam_attr = self.layer_gc.attribute(input_tensor, target=label)
        return gradcam_attr.squeeze().cpu().detach().numpy()

    def _plot_saliency(self, input_tensor, label):
        """
        Compute and process saliency map for visualization.
        
        :param input_tensor: The input image tensor (1, C, H, W).
        :param label: Target class label for the saliency computation.
        :return: Processed saliency map (H, W).
        """
        input_tensor.requires_grad = True
        saliency_map = self.saliency.attribute(input_tensor, target=label)

        # Reduce CHW to HW by averaging across channels
        saliency_map = saliency_map.squeeze().cpu().detach().numpy()
        saliency_map = np.mean(saliency_map, axis=0)  # Average across channels

        return saliency_map


    def explain(self, images, labels, patch, n_samples=5, filename=None):
        """
        Visualize saliency maps and Grad-CAM for a batch of images.

        :param images: Batch of original images.
        :param patch_batch: Batch of patched images.
        :param labels: True labels for the batch.
        :param n_samples: Number of samples to visualize (default: 5).
        :param filename: Optional filename to save the visualization.
        """
        n_samples = min(n_samples, len(images))

        fig, axes = plt.subplots(
            n_samples, 6, figsize=(18, 3 * n_samples)
        )  # 6 columns: Original, Original Saliency, Original Grad-CAM, Patched, Patched Saliency, Patched Grad-CAM
        axes = np.atleast_2d(axes)

        norm_images = images.clone()
        patched_images = self.adv_patch.place_patch(norm_images, patch)

        for i in range(n_samples):
            norm_tensor, patch_tensor, label = images[i], patched_images[i], labels[i]
            norm_img = self._unnormalize_image(norm_tensor)
            patch_img = self._unnormalize_image(patch_tensor)

            # Original Grad-CAM and Saliency
            norm_cam = self._plot_grad_cam(norm_tensor.unsqueeze(0).to(self.device), label)
            norm_sal = self._plot_saliency(norm_tensor.unsqueeze(0).to(self.device), label)

            # Patched Grad-CAM and Saliency
            patch_cam = self._plot_grad_cam(patch_tensor.unsqueeze(0).to(self.device), label)
            patch_sal = self._plot_saliency(patch_tensor.unsqueeze(0).to(self.device), label)

            # Plot each column
            axes[i, 0].imshow(norm_img)
            axes[i, 0].set_title("Original")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(norm_sal, cmap="hot")
            axes[i, 1].set_title("Saliency")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(norm_cam, cmap="jet")
            axes[i, 2].set_title("Grad-CAM")
            axes[i, 2].axis("off")

            axes[i, 3].imshow(patch_img)
            axes[i, 3].set_title("Patched")
            axes[i, 3].axis("off")

            axes[i, 4].imshow(patch_sal, cmap="hot")
            axes[i, 4].set_title("Patched Saliency")
            axes[i, 4].axis("off")

            axes[i, 5].imshow(patch_cam, cmap="jet")
            axes[i, 5].set_title("Patched Grad-CAM")
            axes[i, 5].axis("off")

        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        plt.show()
