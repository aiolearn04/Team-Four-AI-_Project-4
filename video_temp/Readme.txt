# Project Setup Guide

This project uses **Streamlit** for the user interface and requires **FFmpeg** for video processing.

## 1. Install FFmpeg
1. Download the following file:
   ```
   ffmpeg-7.1.1-essentials_build.zip
   ```
2. Extract the ZIP file to:
   ```
   C:\ffmpeg
   ```
3. Add `C:\ffmpeg\bin` to your **system PATH**:
   - Press **Win + R**.
   - Type `SystemPropertiesAdvanced` and press **Enter**.
   - Click **Environment Variables**.
   - Under **System variables**, select `Path` and click **Edit**.
   - Add:
     ```
     C:\ffmpeg\bin
     ```
   - Save the changes.
4. Verify installation:
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
