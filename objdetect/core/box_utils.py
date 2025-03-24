import torch

def box_xyxy_to_cxcywh(boxes):
    """
    Convert boxes from [x1, y1, x2, y2] format to [cx, cy, w, h] format.
    
    Args:
        boxes (Tensor): Boxes of shape (N, 4) in [x1, y1, x2, y2] format
        
    Returns:
        Tensor: Converted boxes in [cx, cy, w, h] format
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)

def box_cxcywh_to_xyxy(boxes):
    """
    Convert boxes from [cx, cy, w, h] format to [x1, y1, x2, y2] format.
    
    Args:
        boxes (Tensor): Boxes of shape (N, 4) in [cx, cy, w, h] format
        
    Returns:
        Tensor: Converted boxes in [x1, y1, x2, y2] format
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)