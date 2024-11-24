import random
import torchvision.models as models

class ModelLoader:
    def __init__(self, model_name="resnet18", class_labels=None, device="cpu"):
        """
        Initialize the PretrainedModelLoader.
        
        :param model_name: Name of the model to load (default: "resnet18").
        :param class_labels: List of class labels or path to a class file (required).
        """
        self.model_name = model_name
        self.device = device
        self.class_labels = self._load_class_labels(class_labels)
        self.model = self._load_model()
    
    def _load_class_labels(self, class_labels):
        """
        Load class labels from a file or directly from a list.
        
        :param class_labels: List of class labels or path to a file.
        :return: List of class labels.
        """
        if class_labels is None:
            raise ValueError("Class labels must be provided as a list or a file path.")

        if isinstance(class_labels, str):  # If a file path is provided
            try:
                with open(class_labels, "r") as f:
                    return [line.strip() for line in f.readlines()]
            except FileNotFoundError:
                raise ValueError(f"Class file '{class_labels}' not found.")
        
        elif isinstance(class_labels, list):  # If a list is provided
            return class_labels
        
        else:
            raise TypeError("class_labels must be a list or a valid file path.")

    def _load_model(self):
        """
        Load the pre-trained model.
        
        :return: A PyTorch model in evaluation mode with frozen weights.
        """
        supported_models = {
            "resnet18": models.resnet18(weights='IMAGENET1K_V1'),
            "resnet50": models.resnet50(weights='IMAGENET1K_V1'),
            "mobilenet_v2": models.mobilenet_v2,
            "efficientnet_b0": models.efficientnet_b0,
        }

        if self.model_name not in supported_models:
            raise ValueError(f"Unsupported model '{self.model_name}'. Supported models: {list(supported_models.keys())}")

        # Load pre-trained model and freeze weights
        model = supported_models[self.model_name].to(self.device)
        model.eval()  # Set model to evaluation mode
        for param in model.parameters():
            param.requires_grad = False

        return model

    def get_model(self):
        """
        Get the pre-trained model.
        
        :return: The pre-trained model.
        """
        return self.model

    def get_class_labels(self):
        """
        Get the class labels.
        
        :return: List of class labels.
        """
        return self.class_labels
