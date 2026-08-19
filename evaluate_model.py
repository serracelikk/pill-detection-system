import argparse
import csv
from pathlib import Path

import yaml
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_data_yaml(path: Path):
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    dataset_root = Path(data.get("path", path.parent)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (path.parent / dataset_root).resolve()
    names = data["names"]
    if isinstance(names, list):
        names = {index: name for index, name in enumerate(names)}
    else:
        names = {int(index): name for index, name in names.items()}

    return data, dataset_root, names


def count_labels(label_path: Path, names: dict[int, str]):
    counts = {name: 0 for name in names.values()}
    if not label_path.exists():
        return counts

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id = int(line.split()[0])
        counts[names[class_id]] += 1

    return counts


def count_predictions(result):
    counts = {name: 0 for name in result.names.values()}
    for class_id in result.boxes.cls:
        counts[result.names[int(class_id)]] += 1
    return counts


def resolve_images_dir(data: dict, dataset_root: Path, split_key: str):
    yaml_key = "val" if split_key == "valid" and "val" in data else split_key
    configured = data.get(yaml_key)
    candidates = []

    if configured:
        configured_path = Path(configured)
        if configured_path.is_absolute():
            candidates.append(configured_path)
        else:
            candidates.append((dataset_root / configured_path).resolve())

    candidates.append(dataset_root / split_key / "images")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="YOLO modelini etiketli dataset klasorunde test eder.")
    parser.add_argument("--model", default="best-4.pt", help="Model dosyasi")
    parser.add_argument("--data", default="data.yaml", help="data.yaml dosyasi")
    parser.add_argument("--split", default="test", choices=["train", "valid", "val", "test"], help="Test edilecek split")
    parser.add_argument("--conf", type=float, default=0.40, help="Guven esigi")
    parser.add_argument("--iou", type=float, default=0.25, help="NMS IoU esigi")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference goruntu boyutu")
    parser.add_argument("--save-images", action="store_true", help="Tahmin gorsellerini runs/detect altina kaydet")
    parser.add_argument("--csv", default="runs/detect/evaluation.csv", help="Sonuc CSV yolu")
    args = parser.parse_args()

    data, dataset_root, names = load_data_yaml(Path(args.data))
    split_key = "valid" if args.split == "val" else args.split
    images_dir = resolve_images_dir(data, dataset_root, split_key)
    labels_dir = dataset_root / split_key / "labels"

    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    model = YOLO(args.model)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(__file__).resolve().parent / "runs" / "detect"

    exact = 0
    total_error = 0
    missed = 0
    extra = 0
    class_error = {name: 0 for name in names.values()}

    rows = []
    for image_path in image_paths:
        result = model(
            str(image_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            save=args.save_images,
            project=str(output_dir),
            name=f"eval_{split_key}",
            exist_ok=True,
            verbose=False,
        )[0]

        label_path = labels_dir / f"{image_path.stem}.txt"
        truth = count_labels(label_path, names)
        pred = count_predictions(result)
        ok = pred == truth
        exact += ok

        row = {
            "image": image_path.name,
            "status": "correct" if ok else "wrong",
            "pred_total": sum(pred.values()),
            "truth_total": sum(truth.values()),
        }

        for name in names.values():
            diff = pred.get(name, 0) - truth.get(name, 0)
            total_error += abs(diff)
            class_error[name] += abs(diff)
            missed += max(0, -diff)
            extra += max(0, diff)
            row[f"pred_{name}"] = pred.get(name, 0)
            row[f"truth_{name}"] = truth.get(name, 0)
            row[f"diff_{name}"] = diff

        rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else ["image", "status"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Model: {args.model}")
    print(f"Split: {split_key}")
    print(f"Images: {len(image_paths)}")
    print(f"Exact correct: {exact}/{len(image_paths)}")
    print(f"Wrong: {len(image_paths) - exact}/{len(image_paths)}")
    print(f"Total absolute error: {total_error}")
    print(f"Missed objects: {missed}")
    print(f"Extra objects: {extra}")
    print(f"Class error: {class_error}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
