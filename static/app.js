/**
 * Bird Sound Locator - Frontend Application
 *
 * Supports:
 * - Single bird detection with directional arrow
 * - Multi-bird mode: AI source separation tracking multiple birds simultaneously
 * - Multi-device triangulation: TDOA-based positioning with 2+ phones
 * - Stereo mic ITD direction estimation
 */

(function () {
    'use strict';

    // Bird colors for multi-bird tracking (up to 6 birds)
    const BIRD_COLORS = [
        '#4ecdc4', '#e74c3c', '#f39c12', '#9b59b6', '#3498db', '#2ecc71',
    ];

    // --- State ---
    const state = {
        isListening: false,
        isScanMode: false,
        isConnected: false,
        enableSeparation: false,
        currentHeading: 0,
        targetHeading: null,
        targetConfidence: 0,
        detectedSpecies: null,
        detections: [],
        activeBirds: [],      // Multi-bird: currently tracked birds
        triangulation: null,  // Multi-device TDOA result
        roomId: null,
        deviceId: null,
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
    const startMultiBtn = $('#start-multi-btn');
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
    const multiDetectionPanel = $('#multi-detection-panel');
    const birdList = $('#bird-list');
    const sourceCount = $('#source-count');
    const triangulationPanel = $('#triangulation-panel');
    const triBearing = $('#tri-bearing');
    const triDistance = $('#tri-distance');
    const triDevices = $('#tri-devices');
    const scanIndicator = $('#scan-indicator');
    const scanBtn = $('#scan-btn');
    const listenBtn = $('#listen-btn');
    const historyBtn = $('#history-btn');
    const historyPanel = $('#history-panel');
    const closeHistory = $('#close-history');
    const historyList = $('#history-list');
    const freqCanvas = $('#freq-canvas');
    const freqCtx = freqCanvas.getContext('2d');
    const roomSection = $('#room-section');
    const createRoomBtn = $('#create-room-btn');
    const joinRoomBtn = $('#join-room-btn');
    const roomIdInput = $('#room-id-input');
    const roomStatus = $('#room-status');

    // --- Initialization ---
    startBtn.addEventListener('click', () => startApp(false));
    startMultiBtn.addEventListener('click', () => {
        state.enableSeparation = true;
        roomSection.classList.remove('hidden');
    });
    createRoomBtn.addEventListener('click', createRoom);
    joinRoomBtn.addEventListener('click', () => {
        const code = roomIdInput.value.trim();
        if (code) {
            state.roomId = code;
            roomStatus.textContent = 'Room code: ' + code + ' (will join on start)';
        }
    });
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

    async function createRoom() {
        try {
            const resp = await fetch('/api/room/create', { method: 'POST' });
            const data = await resp.json();
            state.roomId = data.room_id;
            roomIdInput.value = data.room_id;
            roomStatus.textContent = 'Room created: ' + data.room_id + ' - Share this code!';
        } catch (err) {
            roomStatus.textContent = 'Failed to create room: ' + err.message;
        }
    }

    function startApp(skipSeparation) {
        if (skipSeparation) state.enableSeparation = false;
        requestPermissionsAndStart();
    }

    // --- Permissions & Startup ---
    async function requestPermissionsAndStart() {
        startBtn.disabled = true;
        startMultiBtn.disabled = true;
        permissionStatus.textContent = 'Requesting permissions...';

        try {
            permissionStatus.textContent = 'Requesting microphone access...';
            state.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                    sampleRate: 44100,
                },
            });

            permissionStatus.textContent = 'Requesting camera access...';
            const videoStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
            });
            cameraFeed.srcObject = videoStream;

            if (typeof DeviceOrientationEvent !== 'undefined' &&
                typeof DeviceOrientationEvent.requestPermission === 'function') {
                permissionStatus.textContent = 'Requesting compass access...';
                const response = await DeviceOrientationEvent.requestPermission();
                if (response !== 'granted') {
                    throw new Error('Device orientation permission denied');
                }
            }

            permissionStatus.textContent = 'Getting location...';
            try {
                const pos = await new Promise((resolve, reject) =>
                    navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 5000 })
                );
                state.latitude = pos.coords.latitude;
                state.longitude = pos.coords.longitude;
            } catch {
                console.warn('Geolocation unavailable');
            }

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
            startMultiBtn.disabled = false;
        }
    }

    // --- Audio Capture ---
    function initAudio() {
        state.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 44100,
        });

        const source = state.audioContext.createMediaStreamSource(state.mediaStream);

        state.analyser = state.audioContext.createAnalyser();
        state.analyser.fftSize = 256;
        source.connect(state.analyser);

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

        const totalSamples = state.audioChunkBuffer.reduce((sum, c) => sum + c.pcm.length, 0);
        const merged = new Float32Array(totalSamples);
        let offset = 0;
        for (const chunk of state.audioChunkBuffer) {
            merged.set(chunk.pcm, offset);
            offset += chunk.pcm.length;
        }

        const avgHeading = averageHeading(state.audioChunkBuffer.map((c) => c.heading));
        const int16 = float32ToInt16(merged);
        const base64Data = arrayBufferToBase64(int16.buffer);

        state.wsConnection.send(JSON.stringify({
            type: 'audio_chunk',
            data: base64Data,
            sample_rate: state.audioContext.sampleRate,
            heading: Math.round(avgHeading * 10) / 10,
        }));

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

    // --- Compass ---
    function initCompass() {
        window.addEventListener('deviceorientation', (e) => {
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

        setTimeout(() => {
            if (state.currentHeading === 0) compassHeading.textContent = 'No compass';
        }, 2000);
    }

    // --- WebSocket ---
    function connectWebSocket() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = protocol + '//' + location.host + '/ws/audio';

        setConnectionStatus('connecting');
        const ws = new WebSocket(wsUrl);
        state.wsConnection = ws;

        ws.onopen = () => {
            setConnectionStatus('connected');
            ws.send(JSON.stringify({
                type: 'config',
                latitude: state.latitude,
                longitude: state.longitude,
                enable_separation: state.enableSeparation,
            }));

            // Join room if one was configured
            if (state.roomId) {
                state.deviceId = 'dev-' + Math.random().toString(36).substr(2, 6);
                ws.send(JSON.stringify({
                    type: 'join_room',
                    room_id: state.roomId,
                    device_id: state.deviceId,
                }));
            }
        };

        ws.onmessage = (event) => {
            try {
                handleServerMessage(JSON.parse(event.data));
            } catch (err) {
                console.error('Parse error:', err);
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
        // Single bird detection
        if (msg.type === 'detection' && msg.detections && msg.detections.length > 0) {
            handleSingleDetection(msg);
        }

        // Multi-bird detection (source separation)
        if (msg.type === 'multi_detection' && msg.birds) {
            handleMultiDetection(msg);
        }

        // Multi-device triangulation result
        if (msg.type === 'triangulation' && msg.result) {
            handleTriangulation(msg);
        }

        // Stereo direction
        if (msg.type === 'stereo_direction' && msg.direction) {
            state.targetHeading = msg.direction.heading;
            state.targetConfidence = msg.direction.confidence;
            state.lastDetectionTime = Date.now();
        }

        // Room joined
        if (msg.type === 'room_joined') {
            statusText.textContent = 'Room ' + msg.room_id + ' (' + msg.device_count + ' devices)';
        }

        if (msg.type === 'status') {
            statusText.textContent = msg.message || 'Listening...';
        }
    }

    function handleSingleDetection(msg) {
        const top = msg.detections[0];
        state.detectedSpecies = top.species;
        state.targetConfidence = top.confidence;
        state.lastDetectionTime = Date.now();

        if (msg.direction && msg.direction.heading !== undefined) {
            state.targetHeading = msg.direction.heading;
        }

        speciesName.textContent = top.species;
        speciesConfidence.textContent = Math.round(top.confidence * 100) + '% confidence';

        if (state.targetHeading !== null) {
            const dir = getCardinalDirection(state.targetHeading);
            directionInfo.textContent = 'Direction: ' + dir + ' (' + Math.round(state.targetHeading) + '\u00B0)';
        } else {
            directionInfo.textContent = 'Direction: Scanning...';
        }

        detectionPanel.classList.remove('hidden');
        multiDetectionPanel.classList.add('hidden');

        state.detections.unshift({
            species: top.species,
            confidence: top.confidence,
            heading: state.targetHeading,
            time: new Date(),
        });
        if (state.detections.length > 50) state.detections.pop();
        updateHistoryList();

        if (msg.detections.length > 1) {
            const others = msg.detections.slice(1, 3).map(
                (d) => d.species + ' (' + Math.round(d.confidence * 100) + '%)'
            ).join(', ');
            directionInfo.textContent += ' | Also: ' + others;
        }
    }

    function handleMultiDetection(msg) {
        state.activeBirds = msg.birds;
        state.lastDetectionTime = Date.now();

        detectionPanel.classList.add('hidden');
        multiDetectionPanel.classList.remove('hidden');
        sourceCount.textContent = msg.source_count || msg.birds.length;

        birdList.innerHTML = msg.birds.map((bird, idx) => {
            const color = BIRD_COLORS[idx % BIRD_COLORS.length];
            const headingStr = bird.heading !== undefined
                ? getCardinalDirection(bird.heading) + ' ' + Math.round(bird.heading) + '\u00B0'
                : 'scanning...';
            const freqStr = bird.freq_range
                ? (bird.freq_range[0] / 1000).toFixed(1) + '-' + (bird.freq_range[1] / 1000).toFixed(1) + ' kHz'
                : '';

            return '<div class="bird-item">' +
                '<div class="bird-color-dot" style="background:' + color + '"></div>' +
                '<div class="bird-info">' +
                '<div class="bird-species" style="color:' + color + '">' + escapeHtml(bird.species) + '</div>' +
                '<div class="bird-meta">' + Math.round(bird.confidence * 100) + '% | ' + freqStr + '</div>' +
                '</div>' +
                '<div class="bird-direction">' + headingStr + '</div>' +
                '</div>';
        }).join('');

        // Add all to history
        for (const bird of msg.birds) {
            state.detections.unshift({
                species: bird.species,
                confidence: bird.confidence,
                heading: bird.heading || null,
                time: new Date(),
            });
        }
        if (state.detections.length > 50) state.detections.length = 50;
        updateHistoryList();
    }

    function handleTriangulation(msg) {
        state.triangulation = msg.result;
        state.lastDetectionTime = Date.now();
        triangulationPanel.classList.remove('hidden');

        const r = msg.result;
        const dir = getCardinalDirection(r.bearing);
        triBearing.textContent = dir + ' ' + Math.round(r.bearing) + '\u00B0';
        triDistance.textContent = r.distance > 0 ? 'Distance: ~' + r.distance + 'm' : 'Distance: estimating...';

        const deviceCount = msg.devices ? msg.devices.length : '?';
        triDevices.textContent = deviceCount + ' devices | Confidence: ' + Math.round(r.confidence * 100) + '%';

        // Use triangulation bearing as the target heading
        state.targetHeading = r.bearing;
        state.targetConfidence = r.confidence;
    }

    function getCardinalDirection(heading) {
        const dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
        return dirs[Math.round(heading / 45) % 8];
    }

    // --- UI ---
    function toggleListening() {
        state.isListening = !state.isListening;
        listenBtn.classList.toggle('active', state.isListening);

        if (state.isListening) {
            statusText.textContent = state.enableSeparation ? 'Listening (multi-bird)...' : 'Listening...';
            if (state.audioContext && state.audioContext.state === 'suspended') {
                state.audioContext.resume();
            }
        } else {
            statusText.textContent = 'Paused';
            state.audioChunkBuffer = [];
            detectionPanel.classList.add('hidden');
            multiDetectionPanel.classList.add('hidden');
            triangulationPanel.classList.add('hidden');
        }
    }

    function toggleScanMode() {
        state.isScanMode = !state.isScanMode;
        scanBtn.classList.toggle('active', state.isScanMode);
        scanIndicator.classList.toggle('hidden', !state.isScanMode);
        if (state.isScanMode && !state.isListening) toggleListening();
    }

    function updateHistoryList() {
        if (state.detections.length === 0) {
            historyList.innerHTML = '<div class="empty-history">No detections yet. Start listening to identify birds!</div>';
            return;
        }
        historyList.innerHTML = state.detections.map((d) => {
            const timeStr = d.time.toLocaleTimeString();
            const headingStr = d.heading !== null ? ' | ' + Math.round(d.heading) + '\u00B0' : '';
            return '<div class="history-item">' +
                '<div><div class="species">' + escapeHtml(d.species) + '</div>' +
                '<div class="meta">' + timeStr + headingStr + '</div></div>' +
                '<div class="confidence-badge">' + Math.round(d.confidence * 100) + '%</div></div>';
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

        if (state.lastDetectionTime && Date.now() - state.lastDetectionTime > DETECTION_FADE_MS) {
            state.targetHeading = null;
            state.detectedSpecies = null;
            state.activeBirds = [];
            detectionPanel.classList.add('hidden');
            multiDetectionPanel.classList.add('hidden');
            triangulationPanel.classList.add('hidden');
        }

        if (!state.isListening) return;

        const cx = w / 2;
        const cy = h / 2;

        // Draw center reticle always when listening
        drawReticle(cx, cy);

        // Multi-bird mode: draw individual arrows for each bird
        if (state.activeBirds.length > 0) {
            state.activeBirds.forEach((bird, idx) => {
                if (bird.heading === undefined) return;
                const color = BIRD_COLORS[idx % BIRD_COLORS.length];
                drawBirdArrow(cx, cy, bird.heading, bird.confidence || 0.5, color, bird.species, w, h);
            });
        }
        // Single bird / triangulation arrow
        else if (state.targetHeading !== null) {
            drawBirdArrow(cx, cy, state.targetHeading, state.targetConfidence, '#4ecdc4', state.detectedSpecies, w, h);
        }

        // Scan ring
        if (state.isScanMode && state.scanAmplitudes) {
            drawScanRing(cx, cy, Math.min(w, h) * 0.42);
        }
    }

    function drawReticle(cx, cy) {
        const size = 30;
        ctx.strokeStyle = 'rgba(78, 205, 196, 0.4)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(cx, cy, size, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx - size - 5, cy); ctx.lineTo(cx - size + 10, cy);
        ctx.moveTo(cx + size + 5, cy); ctx.lineTo(cx + size - 10, cy);
        ctx.moveTo(cx, cy - size - 5); ctx.lineTo(cx, cy - size + 10);
        ctx.moveTo(cx, cy + size + 5); ctx.lineTo(cx, cy + size - 10);
        ctx.stroke();
    }

    function drawBirdArrow(cx, cy, targetHeading, confidence, color, label, w, h) {
        let relativeAngle = targetHeading - state.currentHeading;
        while (relativeAngle > 180) relativeAngle -= 360;
        while (relativeAngle < -180) relativeAngle += 360;

        const arrowAngleRad = ((relativeAngle - 90) * Math.PI) / 180;
        const arrowLength = Math.min(w, h) * 0.3;
        const arrowX = cx + Math.cos(arrowAngleRad) * arrowLength;
        const arrowY = cy + Math.sin(arrowAngleRad) * arrowLength;

        // Dashed line
        ctx.strokeStyle = color + (Math.round((0.5 + confidence * 0.5) * 255)).toString(16).padStart(2, '0');
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
        ctx.fillStyle = color;
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

        // Label near arrowhead
        if (label) {
            ctx.fillStyle = color;
            ctx.font = 'bold 11px system-ui';
            ctx.textAlign = 'center';
            const labelX = arrowX + Math.cos(arrowAngleRad) * 20;
            const labelY = arrowY + Math.sin(arrowAngleRad) * 20;
            ctx.fillText(label, labelX, labelY);
        }

        // Highlight when bird is ahead
        if (Math.abs(relativeAngle) < 15) {
            ctx.strokeStyle = '#2ecc71';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(cx, cy, 40, 0, Math.PI * 2);
            ctx.stroke();
            ctx.fillStyle = 'rgba(46, 204, 113, 0.8)';
            ctx.font = 'bold 14px system-ui';
            ctx.textAlign = 'center';
            ctx.fillText('BIRD AHEAD', cx, cy + 55);
        }
    }

    function drawScanRing(cx, cy, radius) {
        if (!state.scanAmplitudes) return;
        const maxAmp = Math.max(...state.scanAmplitudes.map((a) => a.rms), 0.001);
        for (const amp of state.scanAmplitudes) {
            const angleRad = ((amp.heading - state.currentHeading - 90) * Math.PI) / 180;
            const intensity = amp.rms / maxAmp;
            const dotRadius = 3 + intensity * 8;
            ctx.fillStyle = 'rgba(243, 156, 18, ' + (0.3 + intensity * 0.7) + ')';
            ctx.beginPath();
            ctx.arc(cx + Math.cos(angleRad) * radius, cy + Math.sin(angleRad) * radius, dotRadius, 0, Math.PI * 2);
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
            freqCtx.fillStyle = 'hsla(' + hue + ', 70%, 60%, 0.6)';
            freqCtx.fillRect(x, h - barHeight, barWidth - 1, barHeight);
            x += barWidth;
            if (x > w) break;
        }
    }

    function registerServiceWorker() {
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/static/sw.js').catch(() => {});
        }
    }

    updateHistoryList();
})();
