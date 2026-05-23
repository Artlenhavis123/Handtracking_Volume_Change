# Hand Gesture Volume Control

A real-time computer vision application that uses hand gesture recognition to control system volume — no keyboard or mouse required.

## Demo

Control your Mac's volume by pinching your thumb and index finger together or apart. The further apart your fingers, the higher the volume.

## How It Works

1. Webcam captures a live video feed
2. [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) detects hand landmarks in real time (21 keypoints per hand)
3. The Euclidean distance between the **thumb tip** (landmark 4) and **index finger tip** (landmark 8) is calculated
4. That distance is mapped to a volume percentage (0–100%)
5. System volume is updated instantly via `osascript` (only when the value changes)

## Features

- Real-time hand detection and tracking at webcam framerate
- Visual feedback — distance and volume percentage rendered on-screen
- Midpoint overlay showing current gesture reading
- Supports up to 2 hands simultaneously

## Tech Stack

| Library | Purpose |
|---|---|
| Python 3 | Core language |
| OpenCV (`cv2`) | Webcam capture and image rendering |
| MediaPipe Tasks | Hand landmark detection (`HandLandmarker`) |
| osascript | macOS system volume control |
| math | Euclidean distance calculation |

## Requirements

- macOS (volume control uses `osascript`)
- Python 3.x
- Webcam

## Installation

```bash
git clone https://github.com/Artlenhavis123/Handtracking_Volume_Change.git
cd Handtracking_Volume_Change
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

On first run, the MediaPipe `hand_landmarker.task` model (~8 MB) is downloaded automatically into the project directory.

## Usage

- Run the script and allow webcam access
- Hold your hand in front of the camera
- Pinch thumb and index finger **together** → volume decreases
- Pull them **apart** → volume increases
- Distance and volume % are shown on screen in real time
- Press `q` to quit

## Volume Mapping

| Finger Distance | Volume |
|---|---|
| < 50px | 0% (mute) |
| 50–300px | Scaled linearly |
| > 300px | 100% |

## Known Limitations

- Currently macOS only — Windows support via `pycaw` is a planned improvement
- Volume mapping is linear; a smoothing function would reduce abrupt jumps

## Future Improvements

- [ ] Cross-platform support (Windows via `pycaw`, Linux via `amixer`)
- [ ] Gesture smoothing to reduce volume flickering
- [ ] Additional gesture commands (mute toggle, play/pause)
- [ ] Configurable sensitivity settings

## Author

Oliver Havis — [GitHub](https://github.com/Artlenhavis123)
