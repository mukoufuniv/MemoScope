# MemoScope

This repository demonstrates how to use **Grounding DINO** to detect objects based on a text prompt. The project contains utilities for real-time detection from a camera as well as a simple image demo.

## Requirements

The scripts depend on the following Python packages:

- `torch`
- `opencv-python`
- `groundingdino`
- `numpy`

Install the packages using `pip`:

```bash
pip install torch opencv-python groundingdino numpy
```

Download the Grounding DINO configuration and checkpoint files. Refer to the Grounding DINO documentation for the recommended model weights.

## Real-time camera demo

```
python realtime_grounding.py --config <config.py> --weights <model.pth> --prompt "a dog" --camera 0
```

Press `q` to quit the window.

## Image demo

```
python dino_image_demo.py --config <config.py> --weights <model.pth> --prompt "a cat" --image input.jpg --output result.jpg
```

Both scripts require that the Grounding DINO package and model files are available.

