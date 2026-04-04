import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.birdlocator.app',
  appName: 'Bird Sound Locator',
  webDir: '../static',
  server: {
    // ⚠️  IMPORTANT: Set this to your backend server URL before building!
    //
    // Option A — Local development (Mac and iPhone on same WiFi):
    //   Run `ifconfig | grep "inet "` on your Mac to find your IP, then:
    //   url: 'http://192.168.1.xxx:8000/app',
    //
    // Option B — Deployed backend (Railway, etc.):
    //   url: 'https://your-app.up.railway.app/app',
    //
    // The setup script will prompt you for this.
    androidScheme: 'https',
  },
  plugins: {},
  ios: {
    scheme: 'BirdLocator',
    backgroundColor: '#1a1a2e',
  },
};

export default config;
