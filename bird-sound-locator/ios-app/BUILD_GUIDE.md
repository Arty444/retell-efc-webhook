# Building Bird Sound Locator for iPhone 16 Pro Max

## Quick Start (one command)

```bash
cd ios-app
./setup.sh
```

The script handles everything: installs dependencies, configures the backend URL,
sets up the Xcode project, copies the native 4-mic plugin, and opens Xcode.

## Prerequisites

- **macOS** with **Xcode 15+** (from the App Store)
- **Node.js 18+** (`brew install node`)
- **CocoaPods** (`sudo gem install cocoapods`)
- An Apple Developer account (free works for personal device testing)

## What the setup script does

1. Installs Capacitor (npm packages)
2. Asks for your backend server URL
3. Creates the iOS Xcode project (`npx cap add ios`)
4. Copies the `FourMicCapturePlugin` Swift code into the project
5. Adds microphone/camera/location/motion permissions to Info.plist
6. Syncs the web assets into the app bundle
7. Opens the project in Xcode

## After Xcode opens

1. **Signing**: Select your Apple Developer team
   - Click the project in the sidebar → "App" target → "Signing & Capabilities"
   - Choose your team from the dropdown
2. **Connect iPhone** via USB cable
3. **Select your iPhone** as the run target (top toolbar)
4. **Cmd+R** to build and run
5. **Grant all permissions** when the app launches (mic, camera, location, motion)

## Running the backend

The iPhone app connects to your Python backend via WebSocket. You need the
backend running before you open the app.

### Option A: Run locally on the same Mac

```bash
# From the project root (not ios-app)
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Your Mac and iPhone must be on the same WiFi network.
Find your Mac's IP: `ifconfig | grep "inet " | grep -v 127.0.0.1`

### Option B: Deploy to Railway

Push to GitHub and connect to Railway.app. The `Procfile` and `railway.json`
are already configured.

## Why native instead of PWA?

Safari's Web Audio API only exposes 1-2 microphone channels. The iPhone 16 Pro
Max has **4 physical microphones**. This native wrapper uses `AVAudioEngine`
with `.measurement` mode to bypass Apple's beamforming and access raw audio
from all 4 mics for full 3D sound localization via TDOA.

## Architecture

```
iPhone 16 Pro Max Hardware (4 MEMS microphones)
        │
        ▼
AVAudioSession (mode: .measurement)     ← Disables Apple's beamforming
        │
        ▼
AVAudioEngine.inputNode
        │
        ▼
installTap(onBus: 0) → PCM buffers      ← Raw audio, all channels
        │
        ▼
FourMicCapturePlugin.swift               ← Base64-encodes Int16 PCM
        │
        ▼
Capacitor JS bridge
        │
        ▼
native-mic-bridge.js                     ← Routes to WebSocket
        │
        ▼
Python backend (main.py)                 ← Processes audio
        │
        ├─→ audio_direction.py           ← GCC-PHAT TDOA → 3D direction
        ├─→ bird_analyzer.py             ← BirdNET AI → species ID
        └─→ distance_estimator.py        ← 6-method Kalman → distance
        │
        ▼
WebSocket response → AR overlay on phone
```

## Microphone positions (iPhone 16 Pro Max)

```
Mic 0: Bottom-left  (30.0,   0.0, 0.0) mm  ← Left of Lightning/USB-C
Mic 1: Bottom-right (47.6,   0.0, 0.0) mm  ← Right of Lightning/USB-C
Mic 2: Front-top    (38.8, 159.0, 0.0) mm  ← Earpiece area
Mic 3: Rear-camera  (20.0, 150.0, 8.25) mm ← Near camera module
```

## Troubleshooting

**"No 4-mic capture"**: Not all iPhones expose 4 channels. The app falls back
to 2-mic stereo or mono automatically. The 4-mic mode is confirmed working on
iPhone 16 Pro Max with `.measurement` mode.

**"Cannot connect to server"**: Make sure the backend URL is correct and the
server is running. For local dev, both devices must be on the same WiFi.

**"Signing error in Xcode"**: You need an Apple Developer account. A free
account works for personal device testing (app expires after 7 days).
