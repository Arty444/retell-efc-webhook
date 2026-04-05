# Bird Sound Locator

Real-time bird species identification and localization using the iPhone 16 Pro Max's 4-microphone array, BirdNET AI, and AR visualization.

## How It Works

1. **4-Mic Audio Capture** — Uses all 4 microphones on the iPhone 16 Pro Max (with AVAudioEngine in `.measurement` mode to bypass Apple's beamforming) to capture raw directional audio
2. **BirdNET AI Classification** — Audio is streamed to a FastAPI backend that runs [BirdNET](https://github.com/kahst/BirdNET-Analyzer) (via `birdnetlib`) to identify bird species with confidence scores
3. **Direction Estimation** — Time-difference-of-arrival (TDOA) analysis across the 4 microphones determines the heading to the bird (0-360 degrees)
4. **Distance Estimation** — Combines sound intensity, atmospheric absorption modeling, and Doppler analysis to estimate how far away the bird is
5. **AR Overlay** — Results are displayed on a live camera feed with species labels, directional indicators, and distance estimates

## Architecture

```
iPhone 16 Pro Max                    Backend Server
┌──────────────────┐                ┌──────────────────┐
│  4-Mic Capture    │   WebSocket   │  FastAPI + WS     │
│  (AVAudioEngine)  │──────────────▶│  /ws/audio        │
│                   │               │                   │
│  Camera + AR      │◀──────────────│  BirdNET AI       │
│  Overlay          │   Species +   │  Direction Est.   │
│                   │   Direction   │  Distance Est.    │
└──────────────────┘               └──────────────────┘
```

## Quick Start

### Backend Server

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/app` in a browser for the web interface.

### iOS App

```bash
cd ios-app
./setup.sh
```

The setup script will:
- Install Capacitor dependencies
- Configure the backend URL (auto-detects your local IP)
- Sync web assets
- Open Xcode for building to your device

See [ios-app/BUILD_GUIDE.md](ios-app/BUILD_GUIDE.md) for detailed instructions.

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /app` | Web interface |
| `GET /api/room/create` | Create a new listening room |
| `GET /api/room/{id}` | Get room status |
| `WS /ws/audio` | Real-time audio streaming + analysis |

## Tech Stack

- **Backend**: FastAPI, uvicorn, birdnetlib (BirdNET AI), numpy, scipy, scikit-learn
- **iOS**: Capacitor 6.x, AVAudioEngine, Swift (FourMicCapturePlugin)
- **Frontend**: Vanilla JS, WebSocket, Canvas (AR overlay)

## Deployment

Deploy the backend to [Railway](https://railway.app):

```bash
railway up
```

Or any platform that supports Python — `Procfile` and `railway.json` are included.

## License

MIT
