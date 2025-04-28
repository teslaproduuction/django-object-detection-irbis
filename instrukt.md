# Migration Guide: YOLOv5 to YOLOv8

This guide outlines the steps to migrate the Django Object Detection project from YOLOv5 to YOLOv8 and updating to Python 3.11.

## 1. Environment Setup

### 1.1 Install Python 3.11
Download and install Python 3.11 from the [official website](https://www.python.org/downloads/).

### 1.2 Create a new virtual environment
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 1.3 Install updated dependencies
```bash
pip install -r requirements.txt
```

## 2. Code Changes

The following files have been updated to support YOLOv8:

### 2.1 Project-wide Changes
- Updated dependencies in `requirements.txt`
- Updated settings in `config/settings/base.py`
- Renamed YOLOv5 references to YOLOv8 throughout the codebase

### 2.2 Model Detection Changes
- Modified `detectobj/models.py` to use YOLOv8 model options
- Updated `detectobj/views.py` to use the Ultralytics API
- Updated forms in `detectobj/forms.py` for consistency

## 3. YOLOv8 API Usage

The YOLOv8 API is different from YOLOv5. Here are the key differences:

### 3.1 Model Loading
```python
# Old YOLOv5
import yolov5
model = yolov5.load('yolov5s.pt')

# New YOLOv8
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
```

### 3.2 Inference
```python
# Old YOLOv5
results = model(img, size=640)
results_pandas = results.pandas().xyxy[0]

# New YOLOv8
results = model(img, conf=0.45)
# Results are accessed differently
for r in results:
    boxes = r.boxes  # Boxes object for predictions
    # Process boxes...
```

### 3.3 Visualization
```python
# Old YOLOv5
results.render()  # Get images with annotations
img = results.ims[0]  # Get the first image

# New YOLOv8
plotted_img = results[0].plot()  # Get annotated image
```

## 4. Directory Structure

The directory structure remains largely the same, with a few modifications:

```
project/
├── yolov8/           # Changed from yolov5/
│   └── weights/      # Model weights directory
├── media/
│   └── inferenced_image/  # Processed images 
├── apps/
│   ├── detectobj/    # Updated detection code
│   ├── images/       # Image handling
│   ├── modelmanager/ # Model management
│   └── users/        # User management
└── config/           # Project configuration
```

## 5. Database Migration

After updating the codebase:

```bash
python manage.py makemigrations
python manage.py migrate
```

## 6. Testing

Test the application to ensure all features work as expected:

```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000` and try:
1. Creating an ImageSet
2. Uploading images
3. Running detection with YOLOv8 models
4. Verifying the results match expectations

## 7. Troubleshooting

### Common Issues:
- **Model loading errors**: Ensure YOLOv8 models are correctly specified
- **Memory errors**: YOLOv8 may require more memory than YOLOv5
- **CUDA errors**: Update CUDA and PyTorch for compatibility
- **API changes**: YOLOv8 has a different API structure than YOLOv5

If you encounter any issues, check the [Ultralytics documentation](https://docs.ultralytics.com/) for the latest YOLOv8 API details.