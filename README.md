4. Smart Productivity Camera (AI)


In the future, large companies need to precisely monitor their employees’ productivity. A security company wants to build a system that can
determine whether a person is working effectively or not, solely by analyzing images and video.
Challenge goal:
Design an AI system that can:
• Analyze employee behavior in the workplace through live video or images
• Detect different states (productive work, leaving the workstation, sitting idle, moving between desks, etc.)
• Display analytical data in a dashboard
Suggested technologies:
• Computer Vision: to analyze video and frames
• Pose Estimation: to detect body posture (e.g., MediaPipe or OpenPose)
• Action Recognition: to detect activity type from video (e.g., I3D or SlowFast)
• Dashboard frontend: to graphically display results (e.g., Streamlit or Plotly Dash)
Rules and output:
• The output must classify each individual in at least 3 different states
• The system should be able to process a short video (30 seconds to 1 minute) and generate an analytical report at the end
• The dashboard must show the percentage of productive versus unproductive time

# Project Setup Guide

This project uses **Streamlit** for the user interface and requires **FFmpeg** for video processing.

## 1. Install FFmpeg
1. Download the following file:
   ```
   ffmpeg-7.1.1-essentials_build.zip
   ```
   or
   ```
   go to ffmpeg-7.1.1-essentials_build folder and unzip
   ```
3. Extract the ZIP file to:
   ```
   C:\ffmpeg
   ```
4. Add `C:\ffmpeg\bin` to your **system PATH**:
   - Press **Win + R**.
   - Type `SystemPropertiesAdvanced` and press **Enter**.
   - Click **Environment Variables**.
   - Under **System variables**, select `Path` and click **Edit**.
   - Add:
     ```
     C:\ffmpeg\bin
     ```
   - Save the changes.
5. Verify installation:
   ```bash
   ffmpeg -version
   ```

## 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

## 3. Run the Streamlit app
```bash
streamlit run .\usa-ai4.py
```
If the browser does not open automatically, go to:
```
http://localhost:8501
```

## 4. Notes
- Python version should be **3.8 to 3.12**.
- Make sure FFmpeg is installed and added to PATH.
- If FFmpeg errors occur, restart your terminal or computer.

- 🤖Team Four AI _Project 4

🗣Mohsen Keshavarzian
🗣Janyar Rakhshanfar
🗣Mahla Jafarpour
🗣Maryam Tejenjari
