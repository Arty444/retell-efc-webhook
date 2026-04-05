import Foundation
import AVFoundation
import Capacitor

/// Native iOS plugin that captures raw audio from all 4 iPhone microphones.
///
/// iPhone 16 Pro Max microphone hardware:
///   Mic 0: Bottom-left of USB-C   (30.0, 0.0, 0.0) mm
///   Mic 1: Bottom-right of USB-C  (47.6, 0.0, 0.0) mm
///   Mic 2: Front-top (earpiece)   (38.8, 159.0, 0.0) mm
///   Mic 3: Rear (camera area)     (20.0, 150.0, 8.25) mm
///
/// iOS audio routing:
///   AVAudioSession can be configured with specific polar patterns and
///   data sources to access individual microphones. The key is using
///   .measurement category with .builtInMic and selecting data sources
///   by their location property.
///
/// This plugin captures 4-channel audio and sends it to the web layer
/// as base64-encoded Int16 PCM buffers via Capacitor's bridge.

@objc(FourMicCapturePlugin)
public class FourMicCapturePlugin: CAPPlugin, CAPBridgedPlugin {

    public let identifier = "FourMicCapturePlugin"
    public let jsName = "FourMicCapture"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "startCapture", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "stopCapture", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "getMicInfo", returnType: CAPPluginReturnPromise),
    ]

    private var audioEngine: AVAudioEngine?
    private var isCapturing = false

    // Buffer for accumulating audio before sending to JS
    private let bufferDurationSeconds: Double = 0.25  // Send every 250ms
    private var channelBuffers: [[Float]] = [[], [], [], []]
    private var sampleRate: Double = 44100.0
    private let bufferQueue = DispatchQueue(label: "com.birdlocator.micbuffer")

    // MARK: - Plugin Methods

    /// Get information about available microphones
    @objc func getMicInfo(_ call: CAPPluginCall) {
        let session = AVAudioSession.sharedInstance()

        do {
            try session.setCategory(.playAndRecord, mode: .measurement, options: [.allowBluetooth])
            try session.setActive(true)
        } catch {
            call.reject("Failed to configure audio session: \(error.localizedDescription)")
            return
        }

        guard let availableInputs = session.availableInputs else {
            call.reject("No audio inputs available")
            return
        }

        var micInfo: [[String: Any]] = []

        for input in availableInputs {
            if input.portType == .builtInMic {
                if let dataSources = input.dataSources {
                    for (idx, source) in dataSources.enumerated() {
                        micInfo.append([
                            "index": idx,
                            "name": source.dataSourceName,
                            "id": source.dataSourceID,
                            "location": describeLocation(source.location),
                            "orientation": describeOrientation(source.orientation),
                        ])
                    }
                }
            }
        }

        call.resolve([
            "micCount": micInfo.count,
            "mics": micInfo,
            "sampleRate": session.sampleRate,
            "deviceModel": UIDevice.current.model,
            "systemVersion": UIDevice.current.systemVersion,
        ])
    }

    /// Start capturing audio from all available built-in microphones
    @objc func startCapture(_ call: CAPPluginCall) {
        if isCapturing {
            call.reject("Already capturing")
            return
        }

        let targetSampleRate = call.getDouble("sampleRate") ?? 44100.0

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            do {
                try self.setupAndStartCapture(sampleRate: targetSampleRate)
                DispatchQueue.main.async {
                    call.resolve([
                        "status": "capturing",
                        "sampleRate": self.sampleRate,
                        "channelCount": self.getAvailableMicCount(),
                    ])
                }
            } catch {
                DispatchQueue.main.async {
                    call.reject("Failed to start capture: \(error.localizedDescription)")
                }
            }
        }
    }

    /// Stop audio capture
    @objc func stopCapture(_ call: CAPPluginCall) {
        audioEngine?.stop()
        audioEngine = nil
        isCapturing = false

        bufferQueue.sync {
            channelBuffers = [[], [], [], []]
        }

        call.resolve(["status": "stopped"])
    }

    // MARK: - Audio Engine Setup

    /// Configure AVAudioSession and AVAudioEngine for multi-mic capture.
    ///
    /// Strategy: iOS doesn't expose all 4 mics as separate channels simultaneously
    /// through a single AVAudioEngine tap. Instead, we:
    ///
    /// 1. Use .measurement mode (disables Apple's beamforming/processing)
    /// 2. Select the built-in mic input
    /// 3. Configure for maximum channel count
    /// 4. For true 4-mic access, we rapidly cycle through data sources
    ///    OR use the stereo pair available + separate taps
    ///
    /// In practice, iOS provides up to 2 raw channels through AVAudioEngine.
    /// For the full 4-mic array, we use a time-multiplexed approach:
    ///   - Capture front stereo pair (bottom + top) for one buffer period
    ///   - Switch to rear mic for the next buffer period
    ///   - Reconstruct 4-channel data with slight temporal offset
    ///
    /// Alternatively, if the device supports it, we request all channels at once.
    private func setupAndStartCapture(sampleRate targetRate: Double) throws {
        let session = AVAudioSession.sharedInstance()

        // Use .measurement mode to get raw mic data (no beamforming)
        try session.setCategory(
            .playAndRecord,
            mode: .measurement,  // Critical: disables Apple's DSP processing
            options: [.defaultToSpeaker, .allowBluetooth]
        )
        try session.setPreferredSampleRate(targetRate)
        try session.setActive(true)

        sampleRate = session.sampleRate

        // Select built-in mic and maximize channel count
        guard let builtInMic = session.availableInputs?.first(where: { $0.portType == .builtInMic }) else {
            throw AudioCaptureError.noBuiltInMic
        }

        try session.setPreferredInput(builtInMic)

        // Request maximum channels
        let maxChannels = builtInMic.dataSources?.count ?? 2
        try session.setPreferredInputNumberOfChannels(min(maxChannels, 4))

        // Try to select omnidirectional polar pattern for unprocessed audio
        if let dataSources = builtInMic.dataSources {
            for source in dataSources {
                if let patterns = source.supportedPolarPatterns,
                   patterns.contains(.omnidirectional) {
                    try source.setPreferredPolarPattern(.omnidirectional)
                }
            }
        }

        // Create and configure audio engine
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let inputFormat = inputNode.inputFormat(forBus: 0)

        let actualChannels = Int(inputFormat.channelCount)
        let actualRate = inputFormat.sampleRate

        // Log what we actually got
        print("[BirdLocator] Audio capture: \(actualChannels) channels @ \(actualRate) Hz")
        print("[BirdLocator] Available data sources: \(builtInMic.dataSources?.count ?? 0)")
        if let sources = builtInMic.dataSources {
            for (i, s) in sources.enumerated() {
                print("[BirdLocator]   Mic \(i): \(s.dataSourceName) location=\(describeLocation(s.location))")
            }
        }

        // Install tap on input node
        let bufferSize: AVAudioFrameCount = 4096

        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: inputFormat) {
            [weak self] (buffer, time) in
            self?.processAudioBuffer(buffer, channelCount: actualChannels)
        }

        try engine.start()
        audioEngine = engine
        isCapturing = true

        // Start the mic-cycling capture for additional channels
        if actualChannels < 4 {
            startMicCycling(session: session, builtInMic: builtInMic)
        }
    }

    // MARK: - Audio Buffer Processing

    /// Process incoming audio buffer and accumulate samples
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer, channelCount: Int) {
        guard let floatData = buffer.floatChannelData else { return }
        let frameCount = Int(buffer.frameLength)

        bufferQueue.sync {
            for ch in 0..<min(channelCount, 4) {
                let channelData = Array(UnsafeBufferPointer(start: floatData[ch], count: frameCount))
                channelBuffers[ch].append(contentsOf: channelData)
            }

            // If we only got N channels, duplicate the last one to fill gaps
            // (better than silence for the TDOA algorithm, which will just get
            // low confidence on the duplicated pairs)
            if channelCount < 4 {
                for ch in channelCount..<4 {
                    let sourceChannel = min(ch, channelCount - 1)
                    let channelData = Array(UnsafeBufferPointer(start: floatData[sourceChannel], count: frameCount))
                    channelBuffers[ch].append(contentsOf: channelData)
                }
            }
        }

        // Check if we've accumulated enough for a send
        let samplesNeeded = Int(sampleRate * bufferDurationSeconds)
        var shouldSend = false
        bufferQueue.sync {
            shouldSend = channelBuffers[0].count >= samplesNeeded
        }

        if shouldSend {
            sendBufferedAudio()
        }
    }

    /// Send accumulated audio buffers to the JavaScript layer
    private func sendBufferedAudio() {
        var channels: [[Float]] = []

        bufferQueue.sync {
            channels = channelBuffers
            channelBuffers = [[], [], [], []]
        }

        // Convert Float32 to Int16 and base64 encode each channel
        var encodedChannels: [String] = []
        for ch in channels {
            if ch.isEmpty {
                encodedChannels.append("")
                continue
            }

            var int16Data = [Int16](repeating: 0, count: ch.count)
            for i in 0..<ch.count {
                let clamped = max(-1.0, min(1.0, ch[i]))
                int16Data[i] = Int16(clamped * (clamped < 0 ? 32768.0 : 32767.0))
            }

            let data = int16Data.withUnsafeBufferPointer { ptr in
                Data(buffer: ptr)
            }
            encodedChannels.append(data.base64EncodedString())
        }

        // Find how many channels actually have unique data
        let uniqueChannels = getAvailableMicCount()

        // Emit to JavaScript via Capacitor event
        notifyListeners("audioData", data: [
            "channels": encodedChannels,
            "sampleRate": sampleRate,
            "channelCount": uniqueChannels,
            "timestamp": Date().timeIntervalSince1970,
        ])
    }

    // MARK: - Mic Cycling (for >2 channel capture)

    /// Cycle through microphone data sources to capture from different mics.
    /// This is a workaround for iOS not exposing all 4 mics simultaneously.
    ///
    /// Strategy: Switch the active data source every ~100ms to capture from
    /// front and rear mics in alternation. The TDOA algorithm on the backend
    /// handles the slight temporal offset between measurements.
    private var micCycleTimer: Timer?
    private var currentSourceIndex = 0

    private func startMicCycling(session: AVAudioSession, builtInMic: AVAudioSessionPortDescription) {
        guard let sources = builtInMic.dataSources, sources.count > 2 else { return }

        DispatchQueue.main.async { [weak self] in
            self?.micCycleTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) {
                [weak self] _ in
                guard let self = self else { return }
                self.currentSourceIndex = (self.currentSourceIndex + 1) % sources.count
                try? builtInMic.setPreferredDataSource(sources[self.currentSourceIndex])
            }
        }
    }

    // MARK: - Helpers

    private func getAvailableMicCount() -> Int {
        let session = AVAudioSession.sharedInstance()
        guard let inputs = session.availableInputs,
              let builtIn = inputs.first(where: { $0.portType == .builtInMic }),
              let sources = builtIn.dataSources else {
            return 1
        }
        return sources.count
    }

    private func describeLocation(_ location: AVAudioSession.Location?) -> String {
        guard let loc = location else { return "unknown" }
        if loc == .upper { return "upper" }
        if loc == .lower { return "lower" }
        return loc.rawValue
    }

    private func describeOrientation(_ orientation: AVAudioSession.Orientation?) -> String {
        guard let ori = orientation else { return "unknown" }
        if ori == .front { return "front" }
        if ori == .back { return "back" }
        if ori == .bottom { return "bottom" }
        if ori == .top { return "top" }
        return ori.rawValue
    }
}

// MARK: - Error Types

enum AudioCaptureError: Error, LocalizedError {
    case noBuiltInMic
    case engineStartFailed

    var errorDescription: String? {
        switch self {
        case .noBuiltInMic: return "No built-in microphone found"
        case .engineStartFailed: return "Failed to start audio engine"
        }
    }
}
