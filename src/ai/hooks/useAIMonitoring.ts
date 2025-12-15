import { useEffect, useRef, useState, useCallback } from 'react';
import { CheatDetectionModel, DetectionResult } from '../models/CheatDetectionModel';

export interface AIMonitoringState {
  isActive: boolean;
  detectionResult: DetectionResult | null;
  error: string | null;
  isModelLoaded: boolean;
}

export function useAIMonitoring() {
  const [state, setState] = useState<AIMonitoringState>({
    isActive: false,
    detectionResult: null,
    error: null,
    isModelLoaded: false
  });

  const modelRef = useRef<CheatDetectionModel | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const initializeModel = useCallback(async () => {
    try {
      modelRef.current = new CheatDetectionModel();
      await modelRef.current.initialize();
      setState(prev => ({ ...prev, isModelLoaded: true, error: null }));
    } catch (error) {
      setState(prev => ({ 
        ...prev, 
        error: `Failed to initialize AI model: ${error instanceof Error ? error.message : 'Unknown error'}` 
      }));
    }
  }, []);

  const startMonitoring = useCallback(async () => {
    try {
      // Get user media
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: 640, 
          height: 480,
          facingMode: 'user'
        },
        audio: true
      });

      streamRef.current = stream;

      // Create video element
      const video = document.createElement('video');
      video.srcObject = stream;
      video.autoplay = true;
      video.playsInline = true;
      videoRef.current = video;

      // Wait for video to be ready
      await new Promise((resolve) => {
        video.onloadedmetadata = resolve;
      });

      setState(prev => ({ ...prev, isActive: true, error: null }));

      // Start detection loop
      intervalRef.current = setInterval(async () => {
        if (modelRef.current && videoRef.current) {
          try {
            const result = await modelRef.current.detectCheating(videoRef.current);
            setState(prev => ({ ...prev, detectionResult: result }));
          } catch (error) {
            console.error('Detection error:', error);
          }
        }
      }, 1000); // Run detection every second

    } catch (error) {
      setState(prev => ({ 
        ...prev, 
        error: `Failed to start monitoring: ${error instanceof Error ? error.message : 'Unknown error'}` 
      }));
    }
  }, []);

  const stopMonitoring = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current = null;
    }

    setState(prev => ({ ...prev, isActive: false, detectionResult: null }));
  }, []);

  useEffect(() => {
    initializeModel();

    return () => {
      stopMonitoring();
      if (modelRef.current) {
        modelRef.current.dispose();
      }
    };
  }, [initializeModel, stopMonitoring]);

  return {
    ...state,
    startMonitoring,
    stopMonitoring,
    getVideoElement: () => videoRef.current
  };
}