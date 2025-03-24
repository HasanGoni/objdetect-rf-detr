# Object Detection Library

A PyTorch-based object detection library with fastai style.

## Features

- Modular design with clean API inspired by fastai
- Support for multiple object detection architectures:
  - Faster R-CNN
  - RF-DETR (Roboflow Detection Transformer)
- Easy dataset handling for COCO format
- Training and fine-tuning capabilities
- Visualization utilities
- Evaluation metrics

## Installation

```bash
pip install git+https://github.com/HasanGoni/objdetect-rf-detr.git
```

## Usage

### Basic Example

```python
from objdetect.models import create_model
from PIL import Image

# Create a model
model = create_model("faster_rcnn", num_classes=91)  # For COCO dataset

# Load an image
img = Image.open("path/to/image.jpg")

# Make predictions
predictions = model.predict(img)

# Process predictions
for pred in predictions:
    print(f"Detected {len(pred['boxes'])} objects")
    for i in range(len(pred['boxes'])):
        box = pred['boxes'][i].tolist()
        label = pred['labels'][i].item()
        score = pred['scores'][i].item()
        print(f"  Class {label}: {score:.2f} confidence at location {box}")
```

### Using RF-DETR for Object Detection

```python
from objdetect.models import create_model
from PIL import Image

# Create an RF-DETR model 
model = create_model("rfdetr", num_classes=91)  # For COCO dataset

# Load an image
img = Image.open("path/to/image.jpg")

# Make predictions
predictions = model.predict(img)
```

### Fine-tuning RF-DETR on Custom Dataset

```python
from objdetect.models import create_model

# Create an RF-DETR model
model = create_model("rfdetr", num_classes=5)  # 5 custom classes

# Fine-tune on custom dataset
result = model.fine_tune(
    dataset_path='path/to/dataset',  # Path to COCO or YOLO format dataset
    output_dir='./fine_tuned_model',
    epochs=10,
    batch_size=8,
    lr=0.0001,
    img_size=640
)

# Use the fine-tuned model
predictions = model.predict("path/to/image.jpg")
```

## License

Apache 2.0