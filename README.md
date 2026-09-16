# Virtual Game Controller
Control any PC racing game with hand gestures — no controller needed. Just your webcam.

Uses MediaPipe hand tracking + OpenCV to turn your hands into a steering wheel. Make fists to accelerate, open hands to brake, tilt to steer.

## Features

- **Racing Mode** — Two-hand steering wheel with throttle and brake
- **Runner Mode** — One-hand swipe controls for Temple Run-style games
- **Paddle Shifters** — Flick two fingers to shift gears
- **Multi-Camera** — Switch between built-in, USB, and DroidCam cameras live
- **Mirror Toggle** — Flip the camera view instantly
- **Always On Top** — Resizable, movable preview window stays above your game
- **Works with any game that uses arrow keys**

## Quick Start

### Option 1: Download the installer (easiest)

Go to [Releases](../../releases), download `VirtualSteeringWheel-Setup.exe`, and install it. No Python needed.

### Option 2: Run from source

```bash
pip install -r requirements.txt
python src/steering_wheel.py
```

## Controls

| Key | Action |
|-----|--------|
| **ESC** | Quit |
| **M** | Switch Racing / Runner mode |
| **C** | Switch camera |
| **F** | Toggle mirror |
| **E / Q** | Gear up / down (racing mode) |

## Gestures

### Racing Mode (two hands)

| Gesture | Action |
|---------|--------|
| Both fists | Accelerate |
| Both hands open | Brake |
| Tilt hands left/right | Steer |
| Two-finger flick (right hand) | Gear up |
| Two-finger flick (left hand) | Gear down |

### Runner Mode (one hand — press M to switch)

Hold index + middle finger up, then swipe:

| Swipe | Action |
|-------|--------|
| Left / Right | Change lane |
| Up | Jump |
| Down | Slide |

## Settings

Edit the top of `src/steering_wheel.py`:

| Setting | Default | What it does |
|---------|---------|-------------|
| `CAMERA_INDEX` | `1` | Starting camera. Press **C** to cycle |
| `FLIP_CAMERA` | `False` | Starting mirror state. Press **F** to toggle |
| `DEAD_ZONE_DEG` | `12` | Tilt ignored at center (prevents jitter) |
| `CAMERA_FPS` | `60` | Higher = lower input lag |
| `GAME_MODE` | `RACING` | Start in RACING or RUNNER mode |
| `WINDOW_SIZE` | `(960, 720)` | Starting preview size (resizable) |
| `ALWAYS_ON_TOP` | `True` | Keep preview above game window |

> Run your game in **borderless** or **windowed** mode to see the camera overlay.

## Build the Installer Yourself

Requires Windows 10/11, Python 3.10+, and [Inno Setup 6](https://jrsoftware.org/isinfo.php).

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_installer.ps1
```

Output: `packaging\installer\VirtualSteeringWheel-Setup.exe`

## Requirements

- Python 3.9+
- Webcam (built-in, USB, or DroidCam)
- Windows 10/11 or macOS

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera won't open | Press **C** to try other cameras, or change `CAMERA_INDEX` |
| Steering reversed | Press **F** to mirror, or set `FLIP_CAMERA = True` |
| Brake won't trigger | Spread all fingers wide |
| Low FPS | Close other apps or lower camera resolution |

## License

MIT
