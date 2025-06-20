import argparse
import cv2

from grounding_dino_module import GroundingDINOModel
from camera_stream import CameraStream


def main(args: argparse.Namespace) -> None:
    model = GroundingDINOModel(args.config, args.weights)
    stream = CameraStream(index=args.camera)

    while True:
        frame = stream.read()
        if frame is None:
            break

        boxes, scores, phrases = model.predict_boxes(frame, args.prompt)
        annotated = model.draw_boxes(frame, boxes, phrases)

        cv2.imshow("GroundingDINO", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time GroundingDINO demo")
    parser.add_argument("--config", required=True, help="Path to model config")
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--prompt", required=True, help="Text prompt to detect")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()
    main(args)

