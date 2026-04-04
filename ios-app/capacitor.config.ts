import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.birdlocator.app',
  appName: 'Bird Sound Locator',
  // The web app is served from the Python backend
  // In development, point to the local server
  server: {
    // For development: use your local machine's IP
    // url: 'http://192.168.1.x:8000/app',
    // For production: bundle the static files
    androidScheme: 'https',
  },
  // Copy the static web assets from the main project
  webDir: '../static',
  plugins: {
    // No additional Capacitor plugin config needed;
    // FourMicCapturePlugin is a local native plugin
  },
  ios: {
    // Build settings
    scheme: 'BirdLocator',
    // Required for background audio capture
    backgroundColor: '#1a1a2e',
  },
};

export default config;
