import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Test your model")
    parser.add_argument('--image_path', type=str, required=True, help='Path to your image')
    parser.add_argument('--save_path', type=str, required=True, help='Path to the results')
    parser.add_argument('--model_weights', type=str, required=True, help='Path to your trained model weights')
    args = parser.parse_args()
    directory = os.path.dirname(args.save_path)
    dirStr, ext = os.path.splitext(args.save_path)
    filename = dirStr.split("\\")[-1]
    model = YOLO(args.model_weights)
    model.predict(
        source=args.image_path,
        project=directory,
        name=filename,
        show=True,
        save=True
    )

if __name__ == "__main__":
    main()