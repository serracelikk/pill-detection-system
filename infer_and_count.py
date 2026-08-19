from ultralytics import YOLO
import argparse
from pathlib import Path


def count_pills(
    image_path: str,
    model_path: str = "best-4.pt",
    conf: float = 0.40,
    iou: float = 0.25,
    imgsz: int = 640,
):
    """Görüntüdeki toplam hap sayısını ve sınıf bazlı sayıları döndürür."""
    output_dir = Path(__file__).resolve().parent / "runs" / "detect"
    model = YOLO(model_path)
    results = model(
        image_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save=True,
        project=str(output_dir),
        name="detect_inference",
        exist_ok=True,
    )

    total = 0
    class_counts = {}

    for r in results:
        names = r.names if hasattr(r, "names") else model.names
        for cls_id in r.boxes.cls:
            cls_id_int = int(cls_id)
            cls_name = names[cls_id_int]
            total += 1
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    return total, class_counts


def main():
    parser = argparse.ArgumentParser(description="Görüntüdeki hapları tespit edip sayar.")
    parser.add_argument("image_path", help="Test edilecek görüntünün dosya yolu")
    parser.add_argument("--model", default="best-4.pt", help="Model dosyası yolu")
    parser.add_argument("--conf", type=float, default=0.40, help="Güven eşiği")
    parser.add_argument("--iou", type=float, default=0.25, help="Aynı nesneye ait çakışan kutuları eleme eşiği")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference görüntü boyutu")
    args = parser.parse_args()

    total, class_counts = count_pills(
        args.image_path,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )

    parol_count = class_counts.get("parol", 0)
    ecopirin_count = class_counts.get("ecopirin", 0)

    print(f"Kullanılan model: {args.model}")
    print(f"Güven eşiği: {args.conf}")
    print(f"IoU eşiği: {args.iou}")
    print(f"Görüntü boyutu: {args.imgsz}")
    print(f"Görüntüde tespit edilen toplam hap sayısı: {total}")
    print(f"Parol sayısı: {parol_count}")
    print(f"Ecopirin sayısı: {ecopirin_count}")


if __name__ == "__main__":
    main()
