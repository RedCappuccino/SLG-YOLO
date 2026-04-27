import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train a SLG-YOLO model")
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset YAML file')
    parser.add_argument('--epochs', type=int, default=300, help='Epochs for training')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for training')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size for training')
    args = parser.parse_args()
    project = f"runs/train/SLG-YOLO"
    model_path = f'ultralytics/cfg/models/SLG-YOLO/SLG-YOLO.yaml'
    model = YOLO(model_path)
    model.train(
        data=args.dataset,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer='SGD',
        project=project
    )

if __name__ == "__main__":
    main()