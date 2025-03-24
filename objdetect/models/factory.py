from .rfdetr import RFDETR
import torch

def create_model(model_type, num_classes=91, pretrained=True, **kwargs):
    """
    Create a model instance based on type.
    
    Args:
        model_type (str): Type of model to create. Options: "rfdetr", "faster_rcnn"
        num_classes (int): Number of output classes
        pretrained (bool): Whether to load pretrained weights
        **kwargs: Additional arguments for specific model types
        
    Returns:
        torch.nn.Module: An instance of the requested model
    """
    model_type = model_type.lower()
    
    if model_type == "rfdetr":
        model = RFDETR(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_type == "faster_rcnn":
        # Import here to avoid circular imports
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        model = fasterrcnn_resnet50_fpn(pretrained=pretrained, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available types: 'rfdetr', 'faster_rcnn'")
    
    return model