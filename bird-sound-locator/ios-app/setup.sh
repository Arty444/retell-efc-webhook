#!/bin/bash
#
# Bird Sound Locator — iOS App Setup
# Run this on your Mac to build the native iPhone app.
#
# Prerequisites:
#   - Xcode 15+ (from App Store)
#   - Node.js 18+ (brew install node)
#   - CocoaPods (sudo gem install cocoapods)
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "🐦 Bird Sound Locator — iOS Setup"
echo "=================================="
echo ""

# ─── Step 1: Check prerequisites ─────────────────────────────────────────────

check_cmd() {
    if ! command -v "$1" &> /dev/null; then
        echo "ERROR: $1 is not installed."
        echo "  Install it with: $2"
        exit 1
    fi
}

check_cmd node "brew install node"
check_cmd npm "brew install node"
check_cmd pod "sudo gem install cocoapods"
check_cmd xcodebuild "Install Xcode from the App Store"

echo "[OK] All prerequisites found"
echo "  Node: $(node --version)"
echo "  npm:  $(npm --version)"
echo "  pod:  $(pod --version 2>/dev/null || echo 'installed')"
echo ""

# ─── Step 2: Get backend URL ─────────────────────────────────────────────────

echo "The app needs to connect to your Python backend server."
echo ""
echo "Options:"
echo "  1) Local dev — run the server on this Mac (same WiFi as iPhone)"
echo "  2) Deployed  — use a public URL (Railway, ngrok, etc.)"
echo ""

# Try to detect local IP
LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}')

read -p "Enter your backend URL [http://${LOCAL_IP}:8000/app]: " BACKEND_URL
BACKEND_URL="${BACKEND_URL:-http://${LOCAL_IP}:8000/app}"

echo ""
echo "Using backend: $BACKEND_URL"
echo ""

# Write the URL into capacitor.config.json (JSON — no TypeScript needed)
rm -f capacitor.config.ts
cat > capacitor.config.json << JSONEOF
{
  "appId": "com.birdlocator.app",
  "appName": "Bird Sound Locator",
  "webDir": "../static",
  "server": {
    "url": "${BACKEND_URL}",
    "cleartext": true
  },
  "ios": {
    "scheme": "BirdLocator",
    "backgroundColor": "#1a1a2e"
  }
}
JSONEOF

echo "[OK] Backend URL configured"

# ─── Step 3: Install npm dependencies ────────────────────────────────────────

echo ""
echo "Installing Capacitor..."
npm install
echo "[OK] npm packages installed"

# ─── Step 4: Initialize Capacitor ─────────────────────────────────────────────

if [ ! -d "ios" ]; then
    echo ""
    echo "Initializing Capacitor iOS project..."
    npx cap add ios
    echo "[OK] iOS project created"
else
    echo "[OK] iOS project already exists"
fi

# ─── Step 5: Copy native plugin into Xcode project ──────────────────────────

echo ""
echo "Copying 4-mic native plugin..."
PLUGIN_DEST="ios/App/App/Plugins"
mkdir -p "$PLUGIN_DEST"
cp BirdLocator/BirdLocator/Plugins/FourMicCapturePlugin.swift "$PLUGIN_DEST/"
cp BirdLocator/BirdLocator/Plugins/FourMicCapturePlugin.m "$PLUGIN_DEST/"
echo "[OK] FourMicCapturePlugin installed"

# ─── Step 6: Copy Info.plist permissions ─────────────────────────────────────

echo ""
echo "Adding microphone/camera/location/motion permissions..."
PLIST="ios/App/App/Info.plist"
if [ -f "$PLIST" ]; then
    # Add permission descriptions if not already present
    if ! grep -q "NSMicrophoneUsageDescription" "$PLIST"; then
        plutil -insert NSMicrophoneUsageDescription -string "Bird Sound Locator needs microphone access to detect and identify bird sounds using all 4 microphones for 3D sound localization." "$PLIST"
    fi
    if ! grep -q "NSCameraUsageDescription" "$PLIST"; then
        plutil -insert NSCameraUsageDescription -string "Bird Sound Locator uses the camera to show a live viewfinder with AR directional overlays pointing toward detected birds." "$PLIST"
    fi
    if ! grep -q "NSLocationWhenInUseUsageDescription" "$PLIST"; then
        plutil -insert NSLocationWhenInUseUsageDescription -string "Bird Sound Locator uses your location to identify which bird species are likely in your area." "$PLIST"
    fi
    if ! grep -q "NSMotionUsageDescription" "$PLIST"; then
        plutil -insert NSMotionUsageDescription -string "Bird Sound Locator uses motion sensors to map microphone measurements to compass directions for accurate bird localization." "$PLIST"
    fi
    echo "[OK] Permissions added to Info.plist"
else
    echo "[WARN] Info.plist not found — you may need to add permissions manually in Xcode"
fi

# ─── Step 7: Sync web assets ─────────────────────────────────────────────────

echo ""
echo "Syncing web assets to Xcode project..."
npx cap sync ios
echo "[OK] Web assets synced"

# ─── Step 8: Open in Xcode ───────────────────────────────────────────────────

echo ""
echo "=================================="
echo "  SETUP COMPLETE!"
echo "=================================="
echo ""
echo "Opening Xcode now..."
echo ""
echo "In Xcode, you need to:"
echo "  1. Select your Apple Developer team under Signing & Capabilities"
echo "     (Project > App target > Signing & Capabilities > Team)"
echo "  2. Connect your iPhone 16 Pro Max via USB"
echo "  3. Select your iPhone as the build target (top bar)"
echo "  4. Press Cmd+R to build and run"
echo ""
echo "The app will ask for microphone, camera, location, and motion"
echo "permissions when it first launches."
echo ""
echo "Make sure your Python backend is running at:"
echo "  $BACKEND_URL"
echo ""

npx cap open ios
