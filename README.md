# Pill Detection and Counting System

An AI-based object detection system developed to detect and count two different types of pills from images using YOLOv8.

## Project Overview

This project uses a custom-trained YOLOv8 object detection model to detect and count pills in an input image.

The model is trained to recognize two classes:

- Parol
- Ecopirin

For each detected pill, the model generates a bounding box and the system calculates the total number of detected pills.

## Technologies

- Python
- YOLOv8
- Ultralytics
- PyTorch
- OpenCV

## Project Structure

- `train.py` — Model training
- `infer_and_count.py` — Detects and counts pills in an input image
- `evaluate_model.py` — Evaluates model performance
- `data.yaml` — Dataset and class configuration
- `best-4.pt` — Trained YOLOv8 model
- `yolov8n.pt` — YOLOv8 base model

## Classes

The trained model recognizes two classes:

| Class | Description |
|------|-------------|
| 0 | Ecopirin |
| 1 | Parol |

## Requirements

- Python 3.10 or higher
- Ultralytics

Install the required package:

```bash
pip install ultralytics
```

## How to Run

Run the detection and counting script by providing the path of an input image:

```bash
python infer_and_count.py /path/to/image.jpg
```

The program detects the pills in the image, draws bounding boxes around the detected objects, and calculates the total number of detected pills.

## Detection Example

The following images show examples of object detection and classification performed by the model.

<p align="center">
  <img src="https://github.com/user-attachments/assets/6687ea23-0324-470b-9a82-bbd5fddb2fa5" width="300">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/66b00875-0aff-4d55-9ea2-5109b1c63f13" width="300">
</p>

## Notes

The model is trained specifically for Parol and Ecopirin detection. Detection performance may vary depending on image quality, lighting, pill orientation, and overlap between pills.

HEIC images may need to be converted to JPEG or PNG before inference.

## Author

Serra Çelik