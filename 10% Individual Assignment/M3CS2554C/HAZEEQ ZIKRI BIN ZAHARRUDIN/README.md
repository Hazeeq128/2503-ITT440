NAME: HAZEEQ ZIKRI BIN ZAHARRUDIN  
STUDENT ID: 2024650704  
CLASS: M3CS2554C

# PYTORCH

PyTorch is the framework for the image processing and in deep learning-based computer vision. The services that are provided in this framework such as tools for low-level image manipulation and high-level task like object detection and segmentation. PyTorch indeed is the solid tool related to image processing.

## FEATURES:
1. Pixel Level Manipulations (brightness, contrast)
2. Slicing and Cropping
4. Channel Wise Operation (grayscale conversion , channel swapping)
5. Geometric atransform (resize, rotation)
6. Color Adjustment

## INSTALLATION:
1. Basic Installation (CPU)
```bash
pip install torch torchvision
```
2. Verify Installation
```bash
import torch
print (torch.__version__)
```

## BASIC USAGE EXAMPLE
### Basic Image Processing With PyTorch
1. Image Manipulation
```bash
from PIL import Image
import torchvision.transforms as transforms

# Load an image
img = Image.open("example.jpg")

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),                # Resize
    transforms.CenterCrop(224),           # Crop
    transforms.ToTensor(),                # Convert to tensor [0, 1]
    transforms.Normalize(                 # Normalize (mean, std)
        mean=[0.485, 0.456, 0.406],      # ImageNet stats
        std=[0.229, 0.224, 0.225]
    )
])

# Apply transformations
img_tensor = transform(img)  # Shape: [3, 224, 224]
print(img_tensor.shape)
```
2. Applying Filter
```bash
import torch.nn.functional as F

# Define Sobel kernel (edge detection)
kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]])  # Sobel X

# Apply convolution (requires 4D input: [batch, channel, H, W])
edges = F.conv2d(img_tensor.unsqueeze(0), kernel)
```
3. Grayscale Conversion
```bash
grayscale = transforms.Grayscale(num_output_channels=3)  # Keeps 3 channels
gray_img = grayscale(img_tensor)  # Shape: [3, H, W]
```
4. Saturation Adjustment
```bash
# Convert to grayscale first
gray = transforms.Grayscale()(img_tensor)
saturation_factor = 1.5
saturated_img = gray + (img_tensor - gray) * saturation_factor
```
5. Brightness Adjustment
```bash
brightness_factor = 1.2  # >1 = brighter
bright_img = img_tensor * brightness_factor  # Clamp to [0, 1] if needed
```
