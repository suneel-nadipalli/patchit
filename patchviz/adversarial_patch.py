import torch, random
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np

class AdversarialPatch:
    def __init__(self, model, loader, num_channels=3, device="cpu"):
        """
        Initialize the adversarial patch.
        
        :param patch_size: Tuple indicating the size of the patch as a fraction of image dimensions (width, height).
        :param image_size: Tuple indicating the dimensions of the input image (width, height).
        :param num_channels: Number of channels in the image (default: 3 for RGB).
        """
        self.num_channels = num_channels
        self.device = device
        self.model = model
        self.dataset = loader.dataset
        self.dataloader = loader.dataloader
        self.class_to_idx = loader.class_to_idx
        
        # Initialize patch tensor (random values between 0 and 1)

    def patch_forward(self, patch):
        NORM_MEAN = np.array([0.485, 0.456, 0.406])
        NORM_STD = np.array([0.229, 0.224, 0.225])
        TENSOR_MEANS, TENSOR_STD = torch.FloatTensor(NORM_MEAN)[:,None,None], torch.FloatTensor(NORM_STD)[:,None,None]

        """
        Normalize the input patch within the ImageNet min and max values.

        Args:
            patch: The input patch tensor.

        Returns:
            patch: The transformed patch tensor.
        """

        # Map patch values from [-infty,infty] to ImageNet min and max
        patch = (torch.tanh(patch) + 1 - 2 * TENSOR_MEANS) / (2 * TENSOR_STD)
        return patch
    
    def place_patch(self, img, patch):

        """
        Places a given patch on a batch of images.

        Args:
            img: The batch of images.
            patch: The patch tensor to be placed on the images.

        Returns:
            img: The batch of images with the patch placed on random locations.
        """
    
        for i in range(img.shape[0]):
            h_offset = np.random.randint(0,img.shape[2]-patch.shape[1]-1)
            w_offset = np.random.randint(0,img.shape[3]-patch.shape[2]-1)
            img[i,:,h_offset:h_offset+patch.shape[1],w_offset:w_offset+patch.shape[2]] = self.patch_forward(patch)
        return img
    
    def eval_patch(self, patch, target_class):

        """
        Evaluates the performance of a given patch on a validation set.

        Args:
            model: The model to be evaluated.
            patch: The patch tensor to be evaluated.
            val_loader: The validation data loader.
            target_class: The target class for evaluation.

        Returns:
            tuple: A tuple containing two elements:
                - acc: Accuracy of the model predicting the target class on the validation set.
                - top5: Top-5 accuracy of the model predicting the target class on the validation set.
        """

        self.model.eval()

        tp, tp_5, counter = 0., 0., 0.
        with torch.no_grad():
            for img, img_labels in tqdm(self.dataloader, desc="Validating...", leave=False):
                # For stability, place the patch at 4 random locations per image, and average the performance
                for _ in range(4):
                    patch_img = self.place_patch(img, patch)
                    patch_img = patch_img.to(self.device)
                    img_labels = img_labels.to(self.device)
                    pred = self.model(patch_img)
                    # In the accuracy calculation, we need to exclude the images that are of our target class
                    # as we would not "fool" the model into predicting those
                    tp += torch.logical_and(pred.argmax(dim=-1) == target_class, img_labels != target_class).sum()
                    tp_5 += torch.logical_and((pred.topk(5, dim=-1)[1] == target_class).any(dim=-1), img_labels != target_class).sum()
                    counter += (img_labels != target_class).sum()
        acc = tp/counter
        top5 = tp_5/counter
        return acc, top5
        
    def train_patch(self, target_class, patch_size=(48,48), num_epochs=5):
        """
        Trains a patch to fool a given model into predicting a specific target class.

        Args:
            model: The model to be attacked.
            target_class: The target class for the attack.
            patch_size (int): The size of the patch to be trained.
            num_epochs (int): The number of training epochs.

        Returns:
            tuple: A tuple containing two elements:
                - patch.data: The trained patch tensor.
                - dict: A dictionary containing the validation results of the trained patch.
        """

        # Leave a small set of images out to check generalization
        # In most of our experiments, the performance on the hold-out data points
        # was as good as on the training set. Overfitting was little possible due
        # to the small size of the patches.

        train_set, val_set = torch.utils.data.random_split(self.dataset, [0.8, 0.2])
        train_loader = data.DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True, num_workers=8)
        val_loader = data.DataLoader(val_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4)

        patch = nn.Parameter(torch.zeros(3, patch_size[0], patch_size[1]), requires_grad=True)
        optimizer = torch.optim.SGD([patch], lr=1e-1, momentum=0.8)
        loss_module = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(num_epochs):
            t = tqdm(train_loader, leave=False)
            for img, _ in t:
                img = self.place_patch(img, patch)
                img = img.to(self.device)
                pred = self.model(img)
                labels = torch.zeros(img.shape[0], device=pred.device, dtype=torch.long).fill_(target_class)
                loss = loss_module(pred, labels)
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()
                t.set_description(f"Epoch {epoch}, Loss: {loss.item():4.2f}")

        # Final validation
        acc, top5 = self.eval_patch(patch, target_class)

        return patch.data, {"acc": acc.item(), "top5": top5.item()}

    def get_patches(self, class_names, patch_sizes, num_classes=None, num_epochs=5):

        """
        Gets or trains patches for a given list of class names and patch sizes.

        Args:
            class_names: A list of class names for which patches are to be retrieved or trained.
            patch_sizes: A list of patch sizes for which patches are to be retrieved or trained.

        Returns:
            dict: A dictionary containing the evaluation results of patches.
        """

        result_dict = dict()

        if class_names is None:
            if num_classes is None:
                raise ValueError("Either class_names or num_classes must be provided.")
            
            class_names = random.sample(self.class_to_idx.keys(), num_classes)

            print(f"Randomly selected classes: {class_names}")

        # Loop over all classes and patch sizes
        for name in class_names:
            if name not in self.class_to_idx:
                raise ValueError(f"Class name '{name}' not found in class_to_idx.")
            result_dict[name] = dict()
            for patch_size in patch_sizes:
    
                patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
    
                c = self.class_to_idx[name]

                # print(c)
    
                patch, val_results = self.train_patch(target_class=c, patch_size=patch_size, num_epochs=num_epochs)
    
                print(f"Validation results for {name} and {patch_size}:", val_results)

                results = self.eval_patch(patch, target_class=c)

                # Store results and the patches in a dict for better access
                result_dict[name][patch_size[0]] = {
                    "results": results,
                    "patch": patch
                }

        return result_dict
