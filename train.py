from ultralytics import YOLO

def main():
    # Use pre-trained Google Colab model as starting point (transfer learning)
    model = YOLO("best.pt")

    model.train(
        data="data.yaml",
        epochs=20,
        imgsz=512,
        batch=4,
        device="cpu"
    )

if __name__ == "__main__":
    main()