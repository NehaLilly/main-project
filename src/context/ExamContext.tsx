import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface ExamState {
  isActive: boolean;
  suspiciousActivities: SuspiciousActivity[];
  cameraEnabled: boolean;
  microphoneEnabled: boolean;
  tabSwitches: number;
  copyPasteAttempts: number;
  codeSubmissions: CodeSubmission[];
  currentQuestion: number;
  timeRemaining: number;
}

interface SuspiciousActivity {
  id: string;
  type: 'gaze-deviation' | 'tab-switch' | 'copy-paste' | 'audio-anomaly' | 'face-not-detected';
  timestamp: Date;
  severity: 'low' | 'medium' | 'high';
  description: string;
}

interface CodeSubmission {
  id: string;
  code: string;
  language: string;
  timestamp: Date;
  result?: string;
}

interface ExamContextType {
  examState: ExamState;
  startExam: () => void;
  endExam: () => void;
  addSuspiciousActivity: (activity: Omit<SuspiciousActivity, 'id' | 'timestamp'>) => void;
  submitCode: (code: string, language: string) => void;
  updateTimeRemaining: (time: number) => void;
}

const ExamContext = createContext<ExamContextType | undefined>(undefined);

export function ExamProvider({ children }: { children: ReactNode }) {
  const [examState, setExamState] = useState<ExamState>({
    isActive: false,
    suspiciousActivities: [],
    cameraEnabled: false,
    microphoneEnabled: false,
    tabSwitches: 0,
    copyPasteAttempts: 0,
    codeSubmissions: [],
    currentQuestion: 1,
    timeRemaining: 3600, // 1 hour
  });

  const startExam = () => {
    setExamState(prev => ({ ...prev, isActive: true }));
  };

  const endExam = () => {
    setExamState(prev => ({ ...prev, isActive: false }));
  };

  const addSuspiciousActivity = (activity: Omit<SuspiciousActivity, 'id' | 'timestamp'>) => {
    const newActivity: SuspiciousActivity = {
      ...activity,
      id: Date.now().toString(),
      timestamp: new Date(),
    };

    setExamState(prev => ({
      ...prev,
      suspiciousActivities: [...prev.suspiciousActivities, newActivity],
      tabSwitches: activity.type === 'tab-switch' ? prev.tabSwitches + 1 : prev.tabSwitches,
      copyPasteAttempts: activity.type === 'copy-paste' ? prev.copyPasteAttempts + 1 : prev.copyPasteAttempts,
    }));
  };

  const submitCode = (code: string, language: string) => {
    const submission: CodeSubmission = {
      id: Date.now().toString(),
      code,
      language,
      timestamp: new Date(),
    };

    setExamState(prev => ({
      ...prev,
      codeSubmissions: [...prev.codeSubmissions, submission],
    }));
  };

  const updateTimeRemaining = (time: number) => {
    setExamState(prev => ({ ...prev, timeRemaining: time }));
  };

  // Simulate camera and microphone initialization
  useEffect(() => {
    if (examState.isActive) {
      navigator.mediaDevices.getUserMedia({ video: true, audio: true })
        .then(() => {
          setExamState(prev => ({
            ...prev,
            cameraEnabled: true,
            microphoneEnabled: true,
          }));
        })
        .catch(() => {
          addSuspiciousActivity({
            type: 'face-not-detected',
            severity: 'high',
            description: 'Camera or microphone access denied',
          });
        });
    }
  }, [examState.isActive]);

  return (
    <ExamContext.Provider value={{
      examState,
      startExam,
      endExam,
      addSuspiciousActivity,
      submitCode,
      updateTimeRemaining,
    }}>
      {children}
    </ExamContext.Provider>
  );
}

export const useExam = () => {
  const context = useContext(ExamContext);
  if (!context) {
    throw new Error('useExam must be used within an ExamProvider');
  }
  return context;
};