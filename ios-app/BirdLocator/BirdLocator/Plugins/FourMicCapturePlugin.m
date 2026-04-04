#import <Capacitor/Capacitor.h>

// Register the Swift plugin with Capacitor's Objective-C bridge
CAP_PLUGIN(FourMicCapturePlugin, "FourMicCapture",
    CAP_PLUGIN_METHOD(startCapture, CAPPluginReturnPromise);
    CAP_PLUGIN_METHOD(stopCapture, CAPPluginReturnPromise);
    CAP_PLUGIN_METHOD(getMicInfo, CAPPluginReturnPromise);
)
