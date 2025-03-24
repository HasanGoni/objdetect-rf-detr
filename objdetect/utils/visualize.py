import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image

def plot_detections(image, predictions, class_names=None, score_threshold=0.5, figsize=(12, 10)):
    """
    Visualize detection results on an image.
    
    Args:
        image: PIL Image or numpy array
        predictions: Model prediction dictionary with 'boxes', 'scores', 'labels'
        class_names: List of class names for label mapping
        score_threshold: Minimum score to display a detection
        figsize: Figure size
        
    Returns:
        fig, ax: Matplotlib figure and axis
    """
    # Convert to numpy array if PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    
    # Get detection data
    boxes = predictions['boxes'].cpu().detach().numpy()
    scores = predictions['scores'].cpu().detach().numpy()
    labels = predictions['labels'].cpu().detach().numpy()
    
    # Filter by threshold
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # Display each detection
    for box, score, label in zip(boxes, scores, labels):
        # Convert box to plt coordinates
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle patch
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add label text
        label_text = f"{class_names[label] if class_names else label}: {score:.2f}"
        plt.text(x1, y1, label_text, 
                 bbox=dict(facecolor='white', alpha=0.8),
                 fontsize=8, color='black')
    
    plt.axis('off')
    return fig, ax