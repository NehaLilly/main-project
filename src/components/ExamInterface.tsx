import React, { useState, useEffect } from 'react';
import { LogOut, Clock, AlertTriangle, Camera, Mic, Shield } from 'lucide-react';
import { useExam } from '../context/ExamContext';
import CodeEditor from './CodeEditor';
import AIMonitoringPanel from './AIMonitoringPanel';
import QuestionPanel from './QuestionPanel';

interface ExamInterfaceProps {
  onLogout: () => void;
}

export default function ExamInterface({ onLogout }: ExamInterfaceProps) {
  const { examState, startExam, endExam, updateTimeRemaining, addSuspiciousActivity } = useExam();
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    if (!examState.isActive) {
      startExam();
    }

    // Timer countdown
    const timer = setInterval(() => {
      if (examState.timeRemaining > 0) {
        updateTimeRemaining(examState.timeRemaining - 1);
      } else {
        endExam();
      }
    }, 1000);

    // Tab switching detection
    const handleVisibilityChange = () => {
      if (document.hidden && examState.isActive) {
        addSuspiciousActivity({
          type: 'tab-switch',
          severity: 'high',
          description: 'Student switched away from exam tab',
        });
      }
    };

    // Copy-paste detection
    const handleKeyDown = (e: KeyboardEvent) => {
      if (examState.isActive && ((e.ctrlKey || e.metaKey) && (e.key === 'c' || e.key === 'v'))) {
        if (e.target !== document.querySelector('.code-editor')) {
          e.preventDefault();
          addSuspiciousActivity({
            type: 'copy-paste',
            severity: 'medium',
            description: 'Copy/paste attempt detected outside code editor',
          });
        }
      }
    };

    // Fullscreen monitoring
    const handleFullscreenChange = () => {
      const isCurrentlyFullscreen = !!document.fullscreenElement;
      setIsFullscreen(isCurrentlyFullscreen);
      
      if (!isCurrentlyFullscreen && examState.isActive) {
        addSuspiciousActivity({
          type: 'gaze-deviation',
          severity: 'high',
          description: 'Student exited fullscreen mode',
        });
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('fullscreenchange', handleFullscreenChange);

    return () => {
      clearInterval(timer);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, [examState.isActive, examState.timeRemaining]);

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const enterFullscreen = () => {
    document.documentElement.requestFullscreen();
  };

  const handleEndExam = () => {
    endExam();
    onLogout();
  };

  if (!examState.isActive) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <div className="bg-white p-8 rounded-lg shadow-lg text-center">
          <Shield className="w-16 h-16 text-green-600 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Exam Completed</h2>
          <p className="text-gray-600 mb-4">Your responses have been submitted securely.</p>
          <button
            onClick={onLogout}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Return to Login
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top Navigation */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Shield className="w-6 h-6 text-blue-600" />
            <h1 className="text-lg font-semibold text-gray-900">Technical Assessment</h1>
            <span className="text-sm text-gray-500">Question {examState.currentQuestion}/5</span>
          </div>

          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2">
              <Camera className={`w-4 h-4 ${examState.cameraEnabled ? 'text-green-600' : 'text-red-600'}`} />
              <span className={`text-sm ${examState.cameraEnabled ? 'text-green-600' : 'text-red-600'}`}>
                Camera: {examState.cameraEnabled ? 'Active' : 'Inactive'}
              </span>
            </div>

            <div className="flex items-center space-x-2">
              <Mic className={`w-4 h-4 ${examState.microphoneEnabled ? 'text-green-600' : 'text-red-600'}`} />
              <span className={`text-sm ${examState.microphoneEnabled ? 'text-green-600' : 'text-red-600'}`}>
                Mic: {examState.microphoneEnabled ? 'Active' : 'Inactive'}
              </span>
            </div>

            <div className="flex items-center space-x-2">
              <Clock className="w-4 h-4 text-blue-600" />
              <span className="text-sm font-mono text-blue-600">
                {formatTime(examState.timeRemaining)}
              </span>
            </div>

            {examState.suspiciousActivities.length > 0 && (
              <div className="flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 text-orange-600" />
                <span className="text-sm text-orange-600">
                  {examState.suspiciousActivities.length} alerts
                </span>
              </div>
            )}

            <button
              onClick={handleEndExam}
              className="flex items-center space-x-2 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors"
            >
              <LogOut className="w-4 h-4" />
              <span>End Exam</span>
            </button>
          </div>
        </div>
      </div>

      {!isFullscreen && (
        <div className="bg-yellow-100 border-b border-yellow-200 px-6 py-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-yellow-800">
              For security, please enable fullscreen mode
            </span>
            <button
              onClick={enterFullscreen}
              className="bg-yellow-600 text-white px-3 py-1 rounded text-sm hover:bg-yellow-700 transition-colors"
            >
              Enter Fullscreen
            </button>
          </div>
        </div>
      )}

      <div className="flex h-[calc(100vh-80px)]">
        {/* Left Panel - Question and Monitoring */}
        <div className="w-1/3 border-r border-gray-200 bg-white flex flex-col">
          <QuestionPanel />
          <AIMonitoringPanel />
        </div>

        {/* Right Panel - Code Editor */}
        <div className="flex-1 bg-gray-900">
          <CodeEditor />
        </div>
      </div>
    </div>
  );
}