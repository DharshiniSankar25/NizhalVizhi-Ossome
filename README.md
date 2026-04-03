# NizhalVizhi-Ossome
Advanced Multi-Modal Privacy & Surveillance Detection Engine
AIML Track Submission | 2026 Hackathon
NIZHAL VIZHI is a suite designed to identify hidden surveillance threats through Sensor Fusion. By combining Deep Learning, Heuristic Optical Analysis, and RF Signal Intelligence, it provides a 360-degree privacy audit of any environment.

@Technical Architecture
1. AI Vision Layer (YOLOv8)
Each webcam frame is processed through a custom surveillance class filter using the Ultralytics YOLOv8 engine. It specifically identifies Surveillance Form-Factors such as hidden cameras, mobile devices, and unauthorized hardware. Matches are highlighted with high-visibility bounding boxes and probability labels.
2. Precision Optical Glint Analysis
To eliminate noise from environmental lighting, the engine uses a three-stage heuristic gate:
Area Filtering: Matches micro-lens radius (2–10px).
Circularity Filter: A threshold of ≥0.55 ensures only near-perfect circles (actual glass lenses) are flagged.
Enclosing Radius: Validates geometric consistency to exclude elongated surface reflections.
3. RF Signal Intelligence
The scanning core orchestrates multiple system commands to extract real-time SSID data, connection states, and saved profiles.
Block-Based Regex: Slices raw text at SSID anchors to ensure signal strength is mapped correctly, even in multi-BSSID environments.
Threat Classification: Automatically flags hidden SSIDs and hardware signatures from known spy-cam manufacturers (V380, IPC, Tuya, ESP32).
4. Performance Engineering
Asynchronous Threading: A dedicated daemon thread runs RF scans every 8 seconds, decoupling heavy network calls from the video loop to prevent frame-freeze.
Log Management: Implements a 5-second cooldown on repetitive messages to maintain a clean forensic panel.
Session Persistence: Streamlit-optimized state management ensures AI models and threads initialize only once per session.

The Dashboard UI
The interface is built for both high-level verdicts and deep forensic dives:
Privacy Health Score: A weighted formula (50% YOLO, 30% Glint, 20% RF) provides an instant security posture.
Signal Transparency: A live table displays every network found, its source (scan, connection, or saved), and its specific threat level.
Forensic Audit Log: A real-time terminal feed showing the internal logic and processing steps of the engine.

Setup Instructions
Install required dependencies:
pip install streamlit opencv-python-headless ultralytics numpy
Ensure your camera is connected and run the application.
streamlit run app.py

Built for the 2026 Hackathon AIML Track.
