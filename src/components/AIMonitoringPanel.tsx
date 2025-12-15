import React, { useEffect } from 'react';
import { Camera, AlertTriangle, Eye, Brain, Activity, Shield } from 'lucide-react';
import { useAIMonitoring } from '../ai/hooks/useAIMonitoring';
import { useExam } from '../context/ExamContext';

export default function AIMonitoringPanel() {
  const { examState, addSuspiciousActivity } = useExam();
  const { 
    isActive, 
    detectionResult, 
    error, 
    isModelLoaded, 
    startMonitoring, 
    stopMonitoring,
    getVideoElement 
  } = useAIMonitoring();

  useEffect(() => {
    if (examState.isActive && isModelLoaded && !isActive) {
      startMonitoring();
    } else if (!examState.isActive && isActive) {
      stopMonitoring();
    }
  }, [examState.isActive, isModelLoaded, isActive, startMonitoring, stopMonitoring]);

  useEffect(() => {
    if (detectionResult?.suspiciousActivity) {
      const severity = detectionResult.confidence > 0.8 ? 'high' : 
                     detectionResult.confidence > 0.6 ? 'medium' : 'low';
      
      let description = 'AI detected suspicious behavior';
      
      if (!detectionResult.faceDetected) {
        description = 'Face not detected in camera feed';
      } else if (Math.abs(detectionResult.gazeDirection.x) > 0.3 || Math.abs(detectionResult.gazeDirection.y) > 0.3) {
        description = 'Student looking away from screen';
      } else if (Math.abs(detectionResult.headPose.yaw) > 30) {
        description = 'Unusual head movement detected';
      } else if (detectionResult.handsVisible > 2) {
        description = 'Multiple people detected';
      }

      addSuspiciousActivity({
        type: 'gaze-deviation',
        severity,
        description
      });
    }
  }, [detectionResult, addSuspiciousActivity]);

  const renderVideoFeed = () => {
    const video = getVideoElement();
    if (!video) return null;

    return (
      <div className="relative">
        <video
          ref={(el) => {
            if (el && video) {
              el.srcObject = video.srcObject;
            }
          }}
          autoPlay
          playsInline
          muted
          className="w-full h-32 bg-gray-800 rounded object-cover"
        />
        
        {detectionResult && (
          <div className="absolute top-2 left-2 space-y-1">
            <div className={`px-2 py-1 rounded text-xs ${
              detectionResult.faceDetected ? 'bg-green-500 text-white' : 'bg-red-500 text-white'
            }`}>
              {detectionResult.faceDetected ? 'Face Detected' : 'No Face'}
            </div>
            
            <div className={`px-2 py-1 rounded text-xs ${
              detectionResult.suspiciousActivity ? 'bg-red-500 text-white' : 'bg-green-500 text-white'
            }`}>
              {detectionResult.suspiciousActivity ? 'Suspicious' : 'Normal'}
            </div>
          </div>
        )}

        {detectionResult && (
          <div className="absolute top-2 right-2">
            <div className="bg-black bg-opacity-50 text-white px-2 py-1 rounded text-xs">
              Confidence: {(detectionResult.confidence * 100).toFixed(0)}%
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex-1 p-6 overflow-auto">
      <div className="flex items-center space-x-2 mb-4">
        <Brain className="w-5 h-5 text-blue-600" />
        <h3 className="text-lg font-semibold text-gray-900">AI Monitoring</h3>
        <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-green-500' : 'bg-red-500'}`} />
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 text-red-600" />
            <span className="text-sm text-red-800">{error}</span>
          </div>
        </div>
      )}

      {/* Model Status */}
      <div className="bg-gray-50 p-4 rounded-lg mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">AI Model Status</span>
          <div className={`flex items-center space-x-1 ${isModelLoaded ? 'text-green-600' : 'text-orange-600'}`}>
            <Shield className="w-4 h-4" />
            <span className="text-xs">{isModelLoaded ? 'Loaded' : 'Loading...'}</span>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex justify-between">
            <span className="text-gray-600">TensorFlow.js:</span>
            <span className="text-green-600">✓</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">MediaPipe:</span>
            <span className="text-green-600">✓</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Face Detection:</span>
            <span className="text-green-600">✓</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Gaze Tracking:</span>
            <span className="text-green-600">✓</span>
          </div>
        </div>
      </div>

      {/* Live Camera Feed */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">Live Camera Feed</span>
          <div className={`flex items-center space-x-1 ${isActive ? 'text-green-600' : 'text-red-600'}`}>
            <Camera className="w-4 h-4" />
            <span className="text-xs">{isActive ? 'Active' : 'Inactive'}</span>
          </div>
        </div>
        
        {renderVideoFeed()}
      </div>

      {/* AI Analysis Results */}
      {detectionResult && (
        <div className="bg-gray-50 p-4 rounded-lg mb-6">
          <h4 className="text-sm font-medium text-gray-900 mb-3">Real-time Analysis</h4>
          
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Gaze Direction:</span>
              <div className="text-xs text-gray-800 mt-1">
                X: {detectionResult.gazeDirection.x.toFixed(3)}<br/>
                Y: {detectionResult.gazeDirection.y.toFixed(3)}
              </div>
            </div>
            
            <div>
              <span className="text-gray-600">Head Pose:</span>
              <div className="text-xs text-gray-800 mt-1">
                Yaw: {detectionResult.headPose.yaw.toFixed(1)}°<br/>
                Pitch: {detectionResult.headPose.pitch.toFixed(1)}°
              </div>
            </div>
            
            <div>
              <span className="text-gray-600">Hands Visible:</span>
              <div className="text-xs text-gray-800 mt-1">
                {detectionResult.handsVisible}
              </div>
            </div>
            
            <div>
              <span className="text-gray-600">Face Status:</span>
              <div className={`text-xs mt-1 ${detectionResult.faceDetected ? 'text-green-600' : 'text-red-600'}`}>
                {detectionResult.faceDetected ? 'Detected' : 'Not Detected'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Activity Summary */}
      <div className="bg-gray-50 p-4 rounded-lg mb-6">
        <h4 className="text-sm font-medium text-gray-900 mb-3">Activity Summary</h4>
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">Tab Switches</span>
            <span className={`font-medium ${examState.tabSwitches > 0 ? 'text-red-600' : 'text-green-600'}`}>
              {examState.tabSwitches}
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">Copy/Paste Attempts</span>
            <span className={`font-medium ${examState.copyPasteAttempts > 0 ? 'text-red-600' : 'text-green-600'}`}>
              {examState.copyPasteAttempts}
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">AI Alerts</span>
            <span className={`font-medium ${examState.suspiciousActivities.length > 0 ? 'text-orange-600' : 'text-green-600'}`}>
              {examState.suspiciousActivities.length}
            </span>
          </div>
        </div>
      </div>

      {/* Recent Alerts */}
      <div>
        <h4 className="text-sm font-medium text-gray-900 mb-3">Recent AI Alerts</h4>
        <div className="space-y-2 max-h-64 overflow-auto">
          {examState.suspiciousActivities.slice(-10).reverse().map((activity) => (
            <div
              key={activity.id}
              className={`p-3 rounded-lg border-l-4 ${
                activity.severity === 'high' ? 'border-red-500 bg-red-50' :
                activity.severity === 'medium' ? 'border-orange-500 bg-orange-50' :
                'border-yellow-500 bg-yellow-50'
              }`}
            >
              <div className="flex items-start space-x-2">
                <AlertTriangle className={`w-4 h-4 mt-0.5 ${
                  activity.severity === 'high' ? 'text-red-600' :
                  activity.severity === 'medium' ? 'text-orange-600' :
                  'text-yellow-600'
                }`} />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 capitalize">
                    AI Detection
                  </p>
                  <p className="text-xs text-gray-600 mt-1">
                    {activity.description}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    {activity.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            </div>
          ))}
          
          {examState.suspiciousActivities.length === 0 && (
            <div className="text-center py-8">
              <Activity className="w-8 h-8 text-green-600 mx-auto mb-2" />
              <p className="text-sm text-gray-600">No suspicious activities detected</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}