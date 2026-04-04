# Building Bird Sound Locator for iPhone 16 Pro Max

## Why native?

Safari's Web Audio API only exposes 1-2 audio channels from the microphone.
The iPhone 16 Pro Max has 4 physical microphones. This native wrapper uses
AVAudioEngine with `.measurement` mode to bypass Apple's beamforming pipeline
and access raw audio from all 4 mics for full 3D sound localization.

## Prerequisites

- macOS with Xcode 15+
- Node.js 18+
- CocoaPods (`sudo gem install cocoapods`)
- An Apple Developer account (for device deployment)

## Setup

```bash
# From the ios-app directory
cd ios-app

# Install Capacitor
npm init -y
npm install @capacitor/core @capacitor/cli @capacitor/ios

# Initialize Capacitor iOS project
npx cap init "Bird Sound Locator" com.birdlocator.app --web-dir ../static
npx cap add ios

# Copy web assets
npx cap sync ios

# The Swift plugin files are already in place at:
#   BirdLocator/BirdLocator/Plugins/FourMicCapturePlugin.swift
#   BirdLocator/BirdLocator/Plugins/FourMicCapturePlugin.m
#
# Copy them into the Capacitor-generated Xcode project:
cp -r BirdLocator/BirdLocator/Plugins/ ios/App/App/Plugins/

# Open in Xcode
npx cap open ios
```

## Xcode Configuration

1. **Signing**: Select your Apple Developer team under Signing & Capabilities
2. **Bundle ID**: Set to `com.birdlocator.app`
3. **Deployment Target**: iOS 16.0+
4. **Privacy descriptions**: Already set in Info.plist for mic, camera, location, motion
5. **Background Modes**: Audio is enabled for continuous capture

## How the 4-mic capture works

```
iPhone Hardware (4 MEMS mics)
        |
        v
AVAudioSession (mode: .measurement)     <-- Disables Apple's beamforming
        |
        v
AVAudioEngine.inputNode
        |
        v
installTap(onBus: 0) -> PCM buffers     <-- Raw audio from available channels
        |
        v
FourMicCapturePlugin.swift               <-- Encodes to base64 Int16
        |
        v
Capacitor event: "audioData"             <-- Sends to JavaScript
        |
        v
native-mic-bridge.js                     <-- Routes to WebSocket
        |
        v
Python backend (main.py)                 <-- 4-mic TDOA localization
        |
        v
audio_direction.py                       <-- GCC-PHAT + least-squares solver
```

## Testing on device

1. Connect iPhone 16 Pro Max via USB
2. In Xcode: Product > Run (or Cmd+R)
3. Grant all permissions when prompted
4. The app will show "4-MIC NATIVE" badge when native capture is active

## Backend connection

The native app connects to your Python backend via WebSocket. For local
development, update `capacitor.config.ts` with your machine's local IP:

```typescript
server: {
    url: 'http://192.168.1.xxx:8000/app',
}
```

For production, deploy the Python backend (e.g., Railway) and set the URL.
