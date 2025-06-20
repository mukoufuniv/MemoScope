import argparse
import cv2

from grounding_dino_module import GroundingDINOModel


def main(args: argparse.Namespace) -> None:
    model = GroundingDINOModel(args.config, args.weights)
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    boxes, scores, phrases = model.predict_boxes(image, args.prompt)
    annotated = model.draw_boxes(image, boxes, phrases)
    cv2.imwrite(args.output, annotated)
    print(f"Result saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GroundingDINO on a single image")
    parser.add_argument("--config", required=True, help="Path to model config")
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="output.jpg", help="Output image path")
    args = parser.parse_args()
    main(args)

