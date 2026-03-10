# Lightball Effects

Hand-tracked video effects built with MediaPipe and OpenCV.

## Project Layout

- `src/lightball/effects/lightball.py`: light orb effect generator
- `src/lightball/effects/firethrow.py`: fire throw effect generator
- `run_lightball.py`: simple entrypoint for light orb effect
- `run_firethrow.py`: simple entrypoint for fire throw effect
- `assets/models/hand_landmarker.task`: MediaPipe hand landmark model
- `media/input/input.mp4`: source input video
- `media/output/`: generated output videos
- `docs/model_source.txt`: source link for model reference

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_lightball.py
python run_firethrow.py
```

Outputs are written to `media/output/`.
