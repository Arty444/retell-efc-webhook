/**
 * Native 4-Mic Bridge for iOS (Capacitor)
 *
 * When running inside the Capacitor native wrapper on iPhone 16 Pro Max,
 * this module replaces the Web Audio API mic capture with native
 * AVAudioEngine capture that accesses all 4 physical microphones.
 *
 * The native plugin (FourMicCapturePlugin.swift) sends audio data via
 * Capacitor events. This bridge receives those events and forwards them
 * to the existing WebSocket connection in the same format the backend expects.
 *
 * Usage: Include this script after app.js. It auto-detects Capacitor
 * and overrides the mic capture if native access is available.
 */

(function () {
    'use strict';

    // Only activate inside Capacitor native shell
    if (!window.Capacitor || !window.Capacitor.isNativePlatform()) {
        console.log('[NativeMicBridge] Not running in Capacitor, using Web Audio API');
        return;
    }

    console.log('[NativeMicBridge] Capacitor detected, initializing native 4-mic capture');

    var FourMicCapture = window.Capacitor.Plugins.FourMicCapture;
    if (!FourMicCapture) {
        console.warn('[NativeMicBridge] FourMicCapture plugin not found');
        return;
    }

    // Wait for the app to initialize and expose its state
    var checkInterval = setInterval(function () {
        // The main app.js stores state on window.__birdLocatorState for bridge access
        if (window.__birdLocatorState) {
            clearInterval(checkInterval);
            initNativeBridge(window.__birdLocatorState);
        }
    }, 100);

    function initNativeBridge(state) {

        // Query mic hardware info
        FourMicCapture.getMicInfo().then(function (info) {
            console.log('[NativeMicBridge] Mic info:', JSON.stringify(info));
            state.micCount = info.micCount;
            state.hasStereo = info.micCount >= 2;
            state.nativeCapture = true;

            // Update UI badge
            var badge = document.getElementById('mic-mode');
            if (badge) {
                badge.textContent = info.micCount + '-MIC NATIVE';
                badge.className = 'mic-badge';
            }
        }).catch(function (err) {
            console.error('[NativeMicBridge] getMicInfo failed:', err);
        });

        // Listen for native audio data events
        FourMicCapture.addListener('audioData', function (event) {
            if (!state.isListening) return;
            if (!state.wsConnection || state.wsConnection.readyState !== WebSocket.OPEN) return;

            var channels = event.channels;
            var sampleRate = event.sampleRate;
            var channelCount = event.channelCount;

            // Filter out empty channels
            var validChannels = channels.filter(function (ch) { return ch && ch.length > 0; });

            if (validChannels.length >= 4) {
                // Full 4-mic mode
                state.wsConnection.send(JSON.stringify({
                    type: 'audio_4mic',
                    channels: validChannels.slice(0, 4),
                    sample_rate: sampleRate,
                    heading: Math.round(state.currentHeading * 10) / 10,
                    pitch: Math.round(state.devicePitch * 10) / 10,
                    roll: Math.round(state.deviceRoll * 10) / 10,
                }));
            } else if (validChannels.length >= 2) {
                // Stereo fallback
                state.wsConnection.send(JSON.stringify({
                    type: 'audio_stereo',
                    ch1: validChannels[0],
                    ch2: validChannels[1],
                    sample_rate: sampleRate,
                    heading: Math.round(state.currentHeading * 10) / 10,
                    pitch: Math.round(state.devicePitch * 10) / 10,
                    roll: Math.round(state.deviceRoll * 10) / 10,
                }));
            } else if (validChannels.length >= 1) {
                // Mono fallback
                state.wsConnection.send(JSON.stringify({
                    type: 'audio_chunk',
                    data: validChannels[0],
                    sample_rate: sampleRate,
                    heading: Math.round(state.currentHeading * 10) / 10,
                }));
            }
        });

        // Override the toggleListening function to start/stop native capture
        var originalToggle = state._toggleListening;
        state._toggleListening = function () {
            state.isListening = !state.isListening;

            if (state.isListening) {
                FourMicCapture.startCapture({ sampleRate: 44100 }).then(function (result) {
                    console.log('[NativeMicBridge] Capture started:', result);
                    var badge = document.getElementById('mic-mode');
                    if (badge) {
                        badge.textContent = result.channelCount + '-MIC LIVE';
                    }
                }).catch(function (err) {
                    console.error('[NativeMicBridge] Start capture failed:', err);
                    // Fall back to Web Audio
                    if (originalToggle) originalToggle();
                });
            } else {
                FourMicCapture.stopCapture().then(function () {
                    console.log('[NativeMicBridge] Capture stopped');
                });
            }
        };

        console.log('[NativeMicBridge] Bridge initialized, ready for native 4-mic capture');
    }

})();
