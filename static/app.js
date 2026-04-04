/**
 * Bird Sound Locator - Frontend Application
 *
 * Captures audio from the microphone, streams it to the backend for BirdNET
 * classification, tracks device orientation for directional estimation,
 * and renders an AR-style overlay guiding the user toward detected birds.
 */

(function () {
    'use strict';

    // --- State ---
    const state = {
        isListening: false,
        isScanMode: false,
        isConnected: false,
        currentHeading: 0,
        targetHeading: null,
        targetConfidence: 0,
        detectedSpecies: null,
        detections: [],  // history
        latitude: 0,
        longitude: 0,
        audioContext: null,
        mediaStream: null,
        wsConnection: null,
        audioChunkBuffer: [],
        lastSendTime: 0,
    };

    // --- Config ---
    const CHUNK_INTERVAL_MS = 500;
    const SEND_BUFFER_SECONDS = 3;
    const WS_RECONNECT_DELAY = 2000;
    const DETECTION_FADE_MS = 8000;

    // --- DOM Elements ---
    const $ = (sel) => document.querySelector(sel);
    const startScreen = $('#start-screen');
    const appScreen = $('#app-screen');
    const startBtn = $('#start-btn');
    const permissionStatus = $('#permission-status');
    const cameraFeed = $('#camera-feed');
    const overlayCanvas = $('#overlay-canvas');
    const ctx = overlayCanvas.getContext('2d');
    const statusDot = $('#connection-status');
    const statusText = $('#status-text');
    const compassHeading = $('#compass-heading');
    const detectionPanel = $('#detection-panel');
    const speciesName = $('#species-name');
    const speciesConfidence = $('#species-confidence');
    const directionInfo = $('#direction-info');
    const scanIndicator = $('#scan-indicator');
    const scanBtn = $('#scan-btn');
    const listenBtn = $('#listen-btn');
    const historyBtn = $('#history-btn');
    const historyPanel = $('#history-panel');
    const closeHistory = $('#close-history');
    const historyList = $('#history-list');
    const freqCanvas = $('#freq-canvas');
    const freqCtx = freqCanvas.getContext('2d');

    // --- Initialization ---
    startBtn.addEventListener('click', requestPermissionsAndStart);
    listenBtn.addEventListener('click', toggleListening);
    scanBtn.addEventListener('click', toggleScanMode);
    historyBtn.addEventListener('click', () => historyPanel.classList.toggle('hidden'));
    closeHistory.addEventListener('click', () => historyPanel.classList.add('hidden'));

    resizeCanvases();
    window.addEventListener('resize', resizeCanvases);

    function resizeCanvases() {
        overlayCanvas.width = window.innerWidth * devicePixelRatio;
        overlayCanvas.height = window.innerHeight * devicePixelRatio;
        overlayCanvas.style.width = window.innerWidth + 'px';
        overlayCanvas.style.height = window.innerHeight + 'px';
        ctx.scale(devicePixelRatio, devicePixelRatio);

        freqCanvas.width = freqCanvas.parentElement.offsetWidth * devicePixelRatio;
        freqCanvas.height = freqCanvas.parentElement.offsetHeight * devicePixelRatio;
    }

    // --- Permissions & Startup ---
    async function requestPermissionsAndStart() {
        startBtn.disabled = true;
        permissionStatus.textContent = 'Requesting permissions...';

        try {
            // Request microphone
            permissionStatus.textContent = 'Requesting microphone access...';
            state.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                    sampleRate: 44100,
                },
            });

            // Request camera
            permissionStatus.textContent = 'Requesting camera access...';
            const videoStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
            });
            cameraFeed.srcObject = videoStream;

            // Request device orientation (iOS requires explicit permission)
            if (typeof DeviceOrientationEvent !== 'undefined' &&
                typeof DeviceOrientationEvent.requestPermission === 'function') {
                permissionStatus.textContent = 'Requesting compass access...';
                const response = await DeviceOrientationEvent.requestPermission();
                if (response !== 'granted') {
                    throw new Error('Device orientation permission denied');
                }
            }

            // Request geolocation
            permissionStatus.textContent = 'Getting location...';
            try {
                const pos = await new Promise((resolve, reject) =>
                    navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 5000 })
                );
                state.latitude = pos.coords.latitude;
                state.longitude = pos.coords.longitude;
            } catch {
                // Location is optional - BirdNET works without it
                console.warn('Geolocation unavailable, proceeding without location filtering');
            }

            // All permissions granted - switch to app screen
            startScreen.classList.remove('active');
            appScreen.classList.add('active');

            initAudio();
            initCompass();
            connectWebSocket();
            requestAnimationFrame(renderLoop);
            registerServiceWorker();

        } catch (err) {
            permissionStatus.textContent = 'Error: ' + err.message;
            startBtn.disabled = false;
        }
    }

    // --- Audio Capture ---
    function initAudio() {
        state.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 44100,
        });

        const source = state.audioContext.createMediaStreamSource(state.mediaStream);

        // Analyser for frequency visualization
        state.analyser = state.audioContext.createAnalyser();
        state.analyser.fftSize = 256;
        source.connect(state.analyser);

        // ScriptProcessor for capturing PCM data to send to server
        // (AudioWorklet would be preferred but ScriptProcessor is simpler and widely supported)
        const bufferSize = 4096;
        const processor = state.audioContext.createScriptProcessor(bufferSize, 1, 1);

        processor.onaudioprocess = (e) => {
            if (!state.isListening) return;

            const inputData = e.inputBuffer.getChannelData(0);
            const pcmCopy = new Float32Array(inputData);

            state.audioChunkBuffer.push({
                pcm: pcmCopy,
                heading: state.currentHeading,
                timestamp: Date.now(),
            });

            const now = Date.now();
            if (now - state.lastSendTime >= CHUNK_INTERVAL_MS) {
                sendAudioChunk();
                state.lastSendTime = now;
            }
        };

        source.connect(processor);
        processor.connect(state.audioContext.destination);
    }

    function sendAudioChunk() {
        if (!state.wsConnection || state.wsConnection.readyState !== WebSocket.OPEN) return;
        if (state.audioChunkBuffer.length === 0) return;

        // Merge recent buffer into a single chunk
        const totalSamples = state.audioChunkBuffer.reduce((sum, c) => sum + c.pcm.length, 0);
        const merged = new Float32Array(totalSamples);
        let offset = 0;
        for (const chunk of state.audioChunkBuffer) {
            merged.set(chunk.pcm, offset);
            offset += chunk.pcm.length;
        }

        // Get the average heading from chunks
        const avgHeading = averageHeading(state.audioChunkBuffer.map((c) => c.heading));

        // Convert Float32 to Int16 for smaller payload
        const int16 = float32ToInt16(merged);
        const base64 = arrayBufferToBase64(int16.buffer);

        state.wsConnection.send(JSON.stringify({
            type: 'audio_chunk',
            data: base64,
            sample_rate: state.audioContext.sampleRate,
            heading: Math.round(avgHeading * 10) / 10,
        }));

        // Keep a sliding window of SEND_BUFFER_SECONDS
        const cutoff = Date.now() - SEND_BUFFER_SECONDS * 1000;
        state.audioChunkBuffer = state.audioChunkBuffer.filter((c) => c.timestamp > cutoff);
    }

    function float32ToInt16(float32) {
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
            const s = Math.max(-1, Math.min(1, float32[i]));
            int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return int16;
    }

    function arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }

    function averageHeading(headings) {
        if (headings.length === 0) return 0;
        let sinSum = 0, cosSum = 0;
        for (const h of headings) {
            const rad = (h * Math.PI) / 180;
            sinSum += Math.sin(rad);
            cosSum += Math.cos(rad);
        }
        let avg = (Math.atan2(sinSum, cosSum) * 180) / Math.PI;
        return (avg + 360) % 360;
    }

    // --- Compass / Device Orientation ---
    function initCompass() {
        window.addEventListener('deviceorientation', (e) => {
            // webkitCompassHeading for iOS, alpha for Android
            let heading;
            if (e.webkitCompassHeading !== undefined) {
                heading = e.webkitCompassHeading;
            } else if (e.alpha !== null) {
                heading = (360 - e.alpha) % 360;
            } else {
                return;
            }
            state.currentHeading = heading;
            compassHeading.textContent = Math.round(heading) + '\u00B0';
        });

        // Fallback: if no compass events after 2s, show manual heading info
        setTimeout(() => {
            if (state.currentHeading === 0) {
                compassHeading.textContent = 'No compass';
            }
        }, 2000);
    }

    // --- WebSocket Connection ---
    function connectWebSocket() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${location.host}/ws/audio`;

        setConnectionStatus('connecting');

        const ws = new WebSocket(wsUrl);
        state.wsConnection = ws;

        ws.onopen = () => {
            setConnectionStatus('connected');
            // Send config with location
            ws.send(JSON.stringify({
                type: 'config',
                latitude: state.latitude,
                longitude: state.longitude,
            }));
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                handleServerMessage(msg);
            } catch (err) {
                console.error('Failed to parse server message:', err);
            }
        };

        ws.onclose = () => {
            setConnectionStatus('disconnected');
            setTimeout(connectWebSocket, WS_RECONNECT_DELAY);
        };

        ws.onerror = () => {
            setConnectionStatus('disconnected');
            ws.close();
        };
    }

    function setConnectionStatus(status) {
        statusDot.className = 'status-dot ' + status;
        state.isConnected = status === 'connected';
        if (status === 'connected') statusText.textContent = 'Connected';
        else if (status === 'connecting') statusText.textContent = 'Connecting...';
        else statusText.textContent = 'Disconnected';
    }

    function handleServerMessage(msg) {
        if (msg.type === 'detection' && msg.detections && msg.detections.length > 0) {
            const top = msg.detections[0];
            state.detectedSpecies = top.species;
            state.targetConfidence = top.confidence;
            state.lastDetectionTime = Date.now();

            if (msg.direction && msg.direction.heading !== undefined) {
                state.targetHeading = msg.direction.heading;
            }

            // Update detection panel
            speciesName.textContent = top.species;
            speciesConfidence.textContent = `${Math.round(top.confidence * 100)}% confidence`;

            if (state.targetHeading !== null) {
                const dir = getCardinalDirection(state.targetHeading);
                directionInfo.textContent = `Direction: ${dir} (${Math.round(state.targetHeading)}\u00B0)`;
            } else {
                directionInfo.textContent = 'Direction: Scanning...';
            }

            detectionPanel.classList.remove('hidden');

            // Add to history
            state.detections.unshift({
                species: top.species,
                confidence: top.confidence,
                heading: state.targetHeading,
                time: new Date(),
            });
            if (state.detections.length > 50) state.detections.pop();
            updateHistoryList();

            // Show additional detections in the panel
            if (msg.detections.length > 1) {
                const others = msg.detections.slice(1, 3).map(
                    (d) => `${d.species} (${Math.round(d.confidence * 100)}%)`
                ).join(', ');
                directionInfo.textContent += ' | Also: ' + others;
            }

        } else if (msg.type === 'status') {
            statusText.textContent = msg.message || 'Listening...';
        }
    }

    function getCardinalDirection(heading) {
        const dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
        const idx = Math.round(heading / 45) % 8;
        return dirs[idx];
    }

    // --- UI Interactions ---
    function toggleListening() {
        state.isListening = !state.isListening;
        listenBtn.classList.toggle('active', state.isListening);

        if (state.isListening) {
            statusText.textContent = 'Listening...';
            if (state.audioContext && state.audioContext.state === 'suspended') {
                state.audioContext.resume();
            }
        } else {
            statusText.textContent = 'Paused';
            state.audioChunkBuffer = [];
            detectionPanel.classList.add('hidden');
        }
    }

    function toggleScanMode() {
        state.isScanMode = !state.isScanMode;
        scanBtn.classList.toggle('active', state.isScanMode);
        scanIndicator.classList.toggle('hidden', !state.isScanMode);

        if (state.isScanMode && !state.isListening) {
            toggleListening();
        }
    }

    function updateHistoryList() {
        if (state.detections.length === 0) {
            historyList.innerHTML = '<div class="empty-history">No detections yet. Start listening to identify birds!</div>';
            return;
        }

        historyList.innerHTML = state.detections.map((d) => {
            const timeStr = d.time.toLocaleTimeString();
            const headingStr = d.heading !== null ? ` | ${Math.round(d.heading)}\u00B0` : '';
            return `
                <div class="history-item">
                    <div>
                        <div class="species">${escapeHtml(d.species)}</div>
                        <div class="meta">${timeStr}${headingStr}</div>
                    </div>
                    <div class="confidence-badge">${Math.round(d.confidence * 100)}%</div>
                </div>
            `;
        }).join('');
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // --- Rendering ---
    function renderLoop() {
        drawOverlay();
        drawFrequencyVisualizer();
        requestAnimationFrame(renderLoop);
    }

    function drawOverlay() {
        const w = window.innerWidth;
        const h = window.innerHeight;
        ctx.clearRect(0, 0, w, h);

        // Fade out detection after DETECTION_FADE_MS
        if (state.lastDetectionTime && Date.now() - state.lastDetectionTime > DETECTION_FADE_MS) {
            state.targetHeading = null;
            state.detectedSpecies = null;
            detectionPanel.classList.add('hidden');
        }

        if (state.targetHeading === null || !state.isListening) return;

        // Compute relative angle between current heading and target
        let relativeAngle = state.targetHeading - state.currentHeading;
        // Normalize to -180 to 180
        while (relativeAngle > 180) relativeAngle -= 360;
        while (relativeAngle < -180) relativeAngle += 360;

        const cx = w / 2;
        const cy = h / 2;

        // Draw directional arrow
        const arrowAngleRad = ((relativeAngle - 90) * Math.PI) / 180;
        const arrowLength = Math.min(w, h) * 0.3;
        const arrowX = cx + Math.cos(arrowAngleRad) * arrowLength;
        const arrowY = cy + Math.sin(arrowAngleRad) * arrowLength;

        // Arrow line
        ctx.strokeStyle = `rgba(78, 205, 196, ${0.5 + state.targetConfidence * 0.5})`;
        ctx.lineWidth = 3;
        ctx.setLineDash([8, 4]);
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(arrowX, arrowY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Arrowhead
        const headLen = 20;
        const headAngle = Math.atan2(arrowY - cy, arrowX - cx);
        ctx.fillStyle = '#4ecdc4';
        ctx.beginPath();
        ctx.moveTo(arrowX, arrowY);
        ctx.lineTo(
            arrowX - headLen * Math.cos(headAngle - Math.PI / 6),
            arrowY - headLen * Math.sin(headAngle - Math.PI / 6)
        );
        ctx.lineTo(
            arrowX - headLen * Math.cos(headAngle + Math.PI / 6),
            arrowY - headLen * Math.sin(headAngle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fill();

        // Center crosshair / target reticle
        const reticleSize = 30;
        ctx.strokeStyle = 'rgba(78, 205, 196, 0.4)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(cx, cy, reticleSize, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx - reticleSize - 5, cy);
        ctx.lineTo(cx - reticleSize + 10, cy);
        ctx.moveTo(cx + reticleSize + 5, cy);
        ctx.lineTo(cx + reticleSize - 10, cy);
        ctx.moveTo(cx, cy - reticleSize - 5);
        ctx.lineTo(cx, cy - reticleSize + 10);
        ctx.moveTo(cx, cy + reticleSize + 5);
        ctx.lineTo(cx, cy + reticleSize - 10);
        ctx.stroke();

        // If the bird is roughly ahead (within 15 degrees), highlight green
        if (Math.abs(relativeAngle) < 15) {
            ctx.strokeStyle = '#2ecc71';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(cx, cy, reticleSize + 10, 0, Math.PI * 2);
            ctx.stroke();

            ctx.fillStyle = 'rgba(46, 204, 113, 0.8)';
            ctx.font = 'bold 14px system-ui';
            ctx.textAlign = 'center';
            ctx.fillText('BIRD AHEAD', cx, cy + reticleSize + 30);
        }

        // Compass ring around edge showing scan amplitudes
        if (state.isScanMode && state.scanAmplitudes) {
            drawScanRing(cx, cy, Math.min(w, h) * 0.42);
        }
    }

    function drawScanRing(cx, cy, radius) {
        if (!state.scanAmplitudes) return;
        const maxAmp = Math.max(...state.scanAmplitudes.map((a) => a.rms), 0.001);

        for (const amp of state.scanAmplitudes) {
            const angleRad = ((amp.heading - state.currentHeading - 90) * Math.PI) / 180;
            const intensity = amp.rms / maxAmp;
            const dotRadius = 3 + intensity * 8;

            ctx.fillStyle = `rgba(243, 156, 18, ${0.3 + intensity * 0.7})`;
            ctx.beginPath();
            ctx.arc(
                cx + Math.cos(angleRad) * radius,
                cy + Math.sin(angleRad) * radius,
                dotRadius, 0, Math.PI * 2
            );
            ctx.fill();
        }
    }

    function drawFrequencyVisualizer() {
        if (!state.analyser || !state.isListening) {
            freqCtx.clearRect(0, 0, freqCanvas.width, freqCanvas.height);
            return;
        }

        const bufferLength = state.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        state.analyser.getByteFrequencyData(dataArray);

        const w = freqCanvas.width;
        const h = freqCanvas.height;
        freqCtx.clearRect(0, 0, w, h);

        const barWidth = w / bufferLength * 2;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const barHeight = (dataArray[i] / 255) * h;
            const hue = 170 + (dataArray[i] / 255) * 30;
            freqCtx.fillStyle = `hsla(${hue}, 70%, 60%, 0.6)`;
            freqCtx.fillRect(x, h - barHeight, barWidth - 1, barHeight);
            x += barWidth;
            if (x > w) break;
        }
    }

    // --- Service Worker ---
    function registerServiceWorker() {
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/static/sw.js').catch((err) => {
                console.warn('Service worker registration failed:', err);
            });
        }
    }

    // Initial history render
    updateHistoryList();

})();
