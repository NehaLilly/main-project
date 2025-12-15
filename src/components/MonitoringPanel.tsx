import React, { useEffect, useState } from 'react';
import { Camera, AlertTriangle, Eye, Volume2, Activity } from 'lucide-react';
import { useExam } from '../context/ExamContext';

export default function MonitoringPanel() {
  const { examState, addSuspiciousActivity } = useExam();
  const [faceDetected, setFaceDetected] = useState(true);
  const [gazeStatus, setGazeStatus] = useState<'normal' | 'deviated'>('normal');

  // Simulate AI monitoring
  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate random monitoring events
      const random = Math.random();
      
      if (random < 0.1) { // 10% chance of face not detected
        setFaceDetected(false);
        addSuspiciousActivity({
          type: 'face-not-detected',
          severity: 'high',
          description: 'Student face not detected in camera feed',
        });
        setTimeout(() => setFaceDetected(true), 2000);
      }
      
      if (random > 0.85 && random < 0.95) { // 10% chance of gaze deviation
        setGazeStatus('deviated');
        addSuspiciousActivity({
          type: 'gaze-deviation',
          severity: 'medium',
          description: 'Student looking away from screen',
        });
        setTimeout(() => setGazeStatus('normal'), 3000);
      }

      if (random > 0.95) { // 5% chance of audio anomaly
        addSuspiciousActivity({
          type: 'audio-anomaly',
          severity: 'medium',
          description: 'Unusual audio pattern detected',
        });
      }
    }, 10000); // Check every 10 seconds

    return () => clearInterval(interval);
  }, [addSuspiciousActivity]);

  const getSeverityColor = (severity: 'low' | 'medium' | 'high') => {
    switch (severity) {
      case 'low': return 'text-yellow-600 bg-yellow-50';
      case 'medium': return 'text-orange-600 bg-orange-50';
      case 'high': return 'text-red-600 bg-red-50';
    }
  };

  return (
    <div className="flex-1 p-6 overflow-auto">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">AI Monitoring</h3>

      {/* Live Camera Feed Placeholder */}
      <div className="bg-gray-100 rounded-lg p-4 mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">Camera Feed</span>
          <div className={`flex items-center space-x-1 ${faceDetected ? 'text-green-600' : 'text-red-600'}`}>
            <Camera className="w-4 h-4" />
            <span className="text-xs">{faceDetected ? 'Face Detected' : 'No Face'}</span>
          </div>
        </div>
        <div className="bg-gray-800 rounded aspect-video flex items-center justify-center">
          <span className="text-gray-400 text-sm">Live Camera Feed</span>
        </div>
      </div>

      {/* Monitoring Status */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-gray-50 p-3 rounded-lg">
          <div className="flex items-center space-x-2 mb-1">
            <Eye className="w-4 h-4 text-blue-600" />
            <span className="text-sm font-medium">Gaze Tracking</span>
          </div>
          <span className={`text-xs ${gazeStatus === 'normal' ? 'text-green-600' : 'text-orange-600'}`}>
            {gazeStatus === 'normal' ? 'On Screen' : 'Looking Away'}
          </span>
        </div>

        <div className="bg-gray-50 p-3 rounded-lg">
          <div className="flex items-center space-x-2 mb-1">
            <Volume2 className="w-4 h-4 text-blue-600" />
            <span className="text-sm font-medium">Audio</span>
          </div>
          <span className="text-xs text-green-600">Normal</span>
        </div>
      </div>

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
            <span className="text-gray-600">Total Alerts</span>
            <span className={`font-medium ${examState.suspiciousActivities.length > 0 ? 'text-orange-600' : 'text-green-600'}`}>
              {examState.suspiciousActivities.length}
            </span>
          </div>
        </div>
      </div>

      {/* Recent Alerts */}
      <div>
        <h4 className="text-sm font-medium text-gray-900 mb-3">Recent Alerts</h4>
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
                    {activity.type.replace('-', ' ')}
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