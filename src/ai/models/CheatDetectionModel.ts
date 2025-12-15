import * as tf from '@tensorflow/tfjs';
import { FaceMesh } from '@mediapipe/face_mesh';
import { Hands } from '@mediapipe/hands';
import { Pose } from '@mediapipe/pose';

export interface DetectionResult {
  gazeDirection: { x: number; y: number };
  faceDetected: boolean;
  handsVisible: number;
  headPose: { pitch: number; yaw: number; roll: number };
  suspiciousActivity: boolean;
  confidence: number;
}

export class CheatDetectionModel {
  private faceMesh: FaceMesh;
  private hands: Hands;
  private pose: Pose;
  private model: tf.LayersModel | null = null;
  private isInitialized = false;

  constructor() {
    this.faceMesh = new FaceMesh({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    });

    this.hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    this.pose = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
    });

    this.setupMediaPipe();
  }

  private setupMediaPipe() {
    this.faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    this.hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    this.pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      smoothSegmentation: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
  }

  async initialize() {
    try {
      // Load pre-trained model for cheat detection
      // In production, this would be your custom trained model
      this.model = await tf.loadLayersModel('/models/cheat_detection_model.json');
      this.isInitialized = true;
      console.log('Cheat detection model loaded successfully');
    } catch (error) {
      console.warn('Custom model not found, using fallback detection');
      this.isInitialized = true;
    }
  }

  async detectCheating(videoElement: HTMLVideoElement): Promise<DetectionResult> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    ctx.drawImage(videoElement, 0, 0);

    // Process with MediaPipe
    const results = await Promise.all([
      this.processFaceMesh(canvas),
      this.processHands(canvas),
      this.processPose(canvas)
    ]);

    const [faceResults, handResults, poseResults] = results;

    // Calculate gaze direction
    const gazeDirection = this.calculateGazeDirection(faceResults);
    
    // Detect head pose
    const headPose = this.calculateHeadPose(faceResults);
    
    // Count visible hands
    const handsVisible = handResults?.multiHandLandmarks?.length || 0;
    
    // Determine if face is detected
    const faceDetected = !!faceResults?.multiFaceLandmarks?.length;

    // Run through custom model if available
    let suspiciousActivity = false;
    let confidence = 0.5;

    if (this.model) {
      const features = this.extractFeatures(gazeDirection, headPose, handsVisible, faceDetected);
      const prediction = this.model.predict(features) as tf.Tensor;
      const predictionData = await prediction.data();
      suspiciousActivity = predictionData[0] > 0.7;
      confidence = predictionData[0];
      prediction.dispose();
    } else {
      // Fallback rule-based detection
      suspiciousActivity = this.ruleBasedDetection(gazeDirection, headPose, handsVisible, faceDetected);
      confidence = this.calculateConfidence(gazeDirection, headPose, handsVisible, faceDetected);
    }

    return {
      gazeDirection,
      faceDetected,
      handsVisible,
      headPose,
      suspiciousActivity,
      confidence
    };
  }

  private async processFaceMesh(canvas: HTMLCanvasElement): Promise<any> {
    return new Promise((resolve) => {
      this.faceMesh.onResults((results) => resolve(results));
      this.faceMesh.send({ image: canvas });
    });
  }

  private async processHands(canvas: HTMLCanvasElement): Promise<any> {
    return new Promise((resolve) => {
      this.hands.onResults((results) => resolve(results));
      this.hands.send({ image: canvas });
    });
  }

  private async processPose(canvas: HTMLCanvasElement): Promise<any> {
    return new Promise((resolve) => {
      this.pose.onResults((results) => resolve(results));
      this.pose.send({ image: canvas });
    });
  }

  private calculateGazeDirection(faceResults: any): { x: number; y: number } {
    if (!faceResults?.multiFaceLandmarks?.[0]) {
      return { x: 0, y: 0 };
    }

    const landmarks = faceResults.multiFaceLandmarks[0];
    
    // Eye landmarks for gaze estimation
    const leftEye = landmarks[33];
    const rightEye = landmarks[263];
    const noseTip = landmarks[1];

    // Simple gaze estimation based on eye and nose positions
    const gazeX = (leftEye.x + rightEye.x) / 2 - noseTip.x;
    const gazeY = (leftEye.y + rightEye.y) / 2 - noseTip.y;

    return { x: gazeX, y: gazeY };
  }

  private calculateHeadPose(faceResults: any): { pitch: number; yaw: number; roll: number } {
    if (!faceResults?.multiFaceLandmarks?.[0]) {
      return { pitch: 0, yaw: 0, roll: 0 };
    }

    const landmarks = faceResults.multiFaceLandmarks[0];
    
    // Key points for head pose estimation
    const noseTip = landmarks[1];
    const leftEar = landmarks[234];
    const rightEar = landmarks[454];
    const chin = landmarks[175];
    const forehead = landmarks[10];

    // Calculate angles (simplified estimation)
    const yaw = Math.atan2(rightEar.x - leftEar.x, rightEar.z - leftEar.z) * 180 / Math.PI;
    const pitch = Math.atan2(forehead.y - chin.y, forehead.z - chin.z) * 180 / Math.PI;
    const roll = Math.atan2(rightEar.y - leftEar.y, rightEar.x - leftEar.x) * 180 / Math.PI;

    return { pitch, yaw, roll };
  }

  private extractFeatures(
    gazeDirection: { x: number; y: number },
    headPose: { pitch: number; yaw: number; roll: number },
    handsVisible: number,
    faceDetected: boolean
  ): tf.Tensor {
    const features = [
      gazeDirection.x,
      gazeDirection.y,
      headPose.pitch,
      headPose.yaw,
      headPose.roll,
      handsVisible,
      faceDetected ? 1 : 0
    ];

    return tf.tensor2d([features]);
  }

  private ruleBasedDetection(
    gazeDirection: { x: number; y: number },
    headPose: { pitch: number; yaw: number; roll: number },
    handsVisible: number,
    faceDetected: boolean
  ): boolean {
    // Rule-based suspicious activity detection
    const gazeThreshold = 0.3;
    const headPoseThreshold = 30;

    const suspiciousGaze = Math.abs(gazeDirection.x) > gazeThreshold || Math.abs(gazeDirection.y) > gazeThreshold;
    const suspiciousHeadPose = Math.abs(headPose.yaw) > headPoseThreshold || Math.abs(headPose.pitch) > headPoseThreshold;
    const noFaceDetected = !faceDetected;
    const tooManyHands = handsVisible > 2;

    return suspiciousGaze || suspiciousHeadPose || noFaceDetected || tooManyHands;
  }

  private calculateConfidence(
    gazeDirection: { x: number; y: number },
    headPose: { pitch: number; yaw: number; roll: number },
    handsVisible: number,
    faceDetected: boolean
  ): number {
    let confidence = 0.5;

    if (!faceDetected) confidence += 0.3;
    if (Math.abs(gazeDirection.x) > 0.3) confidence += 0.2;
    if (Math.abs(gazeDirection.y) > 0.3) confidence += 0.2;
    if (Math.abs(headPose.yaw) > 30) confidence += 0.15;
    if (handsVisible > 2) confidence += 0.1;

    return Math.min(confidence, 1.0);
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
    }
  }
}