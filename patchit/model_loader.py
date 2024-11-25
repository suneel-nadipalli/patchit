import torch
import torchvision.models as models

class ModelLoader:
    def __init__(self, model_inst, num_classes=None, custom_weights=None, device="cpu"):
        """
        Initialize the ModelLoader with options for custom weights.
        
        :param model_name: Name of the model architecture (default: 'resnet18').
        :param pretrained: Whether to use pretrained weights (default: True).
        :param num_classes: Number of output classes for the model.
        :param custom_weights: Path to a user-provided weight file (optional).
        """
        self.model_inst = model_inst
        self.num_classes = num_classes
        self.custom_weights = custom_weights
        self.device = device
        
        self.model = self._load_model()

    def _load_model(self):
        """
        Load the model architecture and optionally load custom weights.
        """
        # Load the architecture
        model = self.model_inst

        # Adjust the final layer if num_classes is specified
        if self.num_classes and hasattr(model, "fc"):
            model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)
        
        # Load custom weights if provided
        if self.custom_weights:
            model.load_state_dict(torch.load(self.custom_weights, map_location=self.device))
            print(f"Loaded custom weights from {self.custom_weights}.")
        
        return model.to(self.device)

    def predict(self, inputs):
        """
        Perform predictions using the loaded model.
        
        :param inputs: Input tensor to pass through the model.
        :return: Model predictions.
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(inputs)
