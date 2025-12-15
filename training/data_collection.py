#!/usr/bin/env python3
"""
Data Collection Script for Cheat Detection Training
Collects video data with labels for training the AI model
"""

import cv2
import json
import os
import time
from datetime import datetime
from typing import Dict, List
import tkinter as tk
from tkinter import ttk, messagebox
import threading

class DataCollectionGUI:
    """GUI for collecting labeled training data"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Cheat Detection Data Collection")
        self.root.geometry("800x600")
        
        self.is_recording = False
        self.current_label = "normal"
        self.video_writer = None
        self.cap = None
        self.labels = []
        self.start_time = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.setup_gui()
        self.setup_camera()
    
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video display frame
        self.video_frame = ttk.Frame(main_frame)
        self.video_frame.grid(row=0, column=0, columnspan=2, pady=10)
        
        self.video_label = ttk.Label(self.video_frame, text="Camera Feed")
        self.video_label.pack()
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Recording controls
        self.record_button = ttk.Button(control_frame, text="Start Recording", 
                                       command=self.toggle_recording)
        self.record_button.grid(row=0, column=0, padx=5)
        
        self.status_label = ttk.Label(control_frame, text="Ready to record")
        self.status_label.grid(row=0, column=1, padx=10)
        
        # Label selection
        label_frame = ttk.LabelFrame(main_frame, text="Current Behavior", padding="10")
        label_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        self.label_var = tk.StringVar(value="normal")
        
        behaviors = [
            ("Normal Behavior", "normal"),
            ("Looking Away", "looking_away"),
            ("Using Phone", "using_phone"),
            ("Reading Notes", "reading_notes"),
            ("Talking to Someone", "talking"),
            ("Multiple People", "multiple_people"),
            ("No Face Visible", "no_face"),
            ("Suspicious Movement", "suspicious_movement")
        ]
        
        for i, (text, value) in enumerate(behaviors):
            ttk.Radiobutton(label_frame, text=text, variable=self.label_var, 
                           value=value, command=self.on_label_change).grid(
                           row=i//2, column=i%2, sticky=tk.W, padx=10, pady=2)
        
        # Session info
        info_frame = ttk.LabelFrame(main_frame, text="Session Info", padding="10")
        info_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Label(info_frame, text=f"Session ID: {self.session_id}").grid(row=0, column=0, sticky=tk.W)
        self.duration_label = ttk.Label(info_frame, text="Duration: 00:00")
        self.duration_label.grid(row=1, column=0, sticky=tk.W)
        self.labels_count_label = ttk.Label(info_frame, text="Labels: 0")
        self.labels_count_label.grid(row=2, column=0, sticky=tk.W)
        
        # Instructions
        instructions = """
Instructions:
1. Click 'Start Recording' to begin data collection
2. Select the appropriate behavior label during recording
3. Change labels in real-time as behavior changes
4. Click 'Stop Recording' when finished
5. Data will be saved automatically
        """
        
        ttk.Label(main_frame, text=instructions, justify=tk.LEFT).grid(
            row=4, column=0, columnspan=2, pady=10)
    
    def setup_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.update_video_feed()
    
    def update_video_feed(self):
        """Update video feed display"""
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Convert frame for tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (480, 360))
                
                # Convert to PhotoImage
                from PIL import Image, ImageTk
                image = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(image)
                
                self.video_label.configure(image=photo)
                self.video_label.image = photo
                
                # Write frame if recording
                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(frame)
        
        # Schedule next update
        self.root.after(33, self.update_video_feed)  # ~30 FPS
    
    def toggle_recording(self):
        """Start or stop recording"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording video and labels"""
        # Create output directory
        output_dir = f"training_data/{self.session_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(output_dir, f"{self.session_id}.mp4")
        self.video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        
        # Initialize recording state
        self.is_recording = True
        self.start_time = time.time()
        self.labels = []
        
        # Add initial label
        self.add_label_event()
        
        # Update UI
        self.record_button.configure(text="Stop Recording")
        self.status_label.configure(text="Recording...")
        
        # Start duration timer
        self.update_duration()
    
    def stop_recording(self):
        """Stop recording and save data"""
        self.is_recording = False
        
        # Add final label event
        self.add_label_event()
        
        # Release video writer
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        # Save labels
        output_dir = f"training_data/{self.session_id}"
        labels_path = os.path.join(output_dir, f"{self.session_id}_labels.json")
        
        with open(labels_path, 'w') as f:
            json.dump(self.labels, f, indent=2)
        
        # Update UI
        self.record_button.configure(text="Start Recording")
        self.status_label.configure(text="Recording saved")
        
        messagebox.showinfo("Success", f"Recording saved to {output_dir}")
    
    def on_label_change(self):
        """Handle label change during recording"""
        if self.is_recording:
            self.add_label_event()
    
    def add_label_event(self):
        """Add a label event with timestamp"""
        if self.start_time is not None:
            current_time = time.time()
            timestamp = current_time - self.start_time
            
            # End previous label if exists
            if self.labels:
                self.labels[-1]['end_time'] = timestamp
            
            # Add new label
            label_event = {
                'start_time': timestamp,
                'end_time': None,  # Will be set when label changes or recording stops
                'behavior': self.label_var.get(),
                'is_cheating': self.label_var.get() != 'normal'
            }
            
            self.labels.append(label_event)
            
            # Update labels count
            self.labels_count_label.configure(text=f"Labels: {len(self.labels)}")
    
    def update_duration(self):
        """Update recording duration display"""
        if self.is_recording and self.start_time is not None:
            duration = time.time() - self.start_time
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            self.duration_label.configure(text=f"Duration: {minutes:02d}:{seconds:02d}")
            
            # Schedule next update
            self.root.after(1000, self.update_duration)
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_recording:
            if messagebox.askokcancel("Quit", "Recording in progress. Stop recording and quit?"):
                self.stop_recording()
            else:
                return
        
        if self.cap is not None:
            self.cap.release()
        
        self.root.destroy()
    
    def run(self):
        """Run the data collection GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

def create_sample_labels():
    """Create sample label file for demonstration"""
    sample_labels = [
        {
            "start_time": 0.0,
            "end_time": 30.0,
            "behavior": "normal",
            "is_cheating": False
        },
        {
            "start_time": 30.0,
            "end_time": 45.0,
            "behavior": "looking_away",
            "is_cheating": True
        },
        {
            "start_time": 45.0,
            "end_time": 60.0,
            "behavior": "using_phone",
            "is_cheating": True
        },
        {
            "start_time": 60.0,
            "end_time": 90.0,
            "behavior": "normal",
            "is_cheating": False
        }
    ]
    
    os.makedirs("training_data/sample", exist_ok=True)
    with open("training_data/sample/sample_labels.json", 'w') as f:
        json.dump(sample_labels, f, indent=2)
    
    print("Sample labels created at training_data/sample/sample_labels.json")

if __name__ == "__main__":
    print("Cheat Detection Data Collection Tool")
    print("====================================")
    
    choice = input("1. Run data collection GUI\n2. Create sample labels\nChoice (1/2): ")
    
    if choice == "1":
        app = DataCollectionGUI()
        app.run()
    elif choice == "2":
        create_sample_labels()
    else:
        print("Invalid choice")