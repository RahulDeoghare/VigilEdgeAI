# Vehicle Monitoring System (VMS)

A computer vision-based vehicle monitoring system that uses YOLO models for vehicle detection and ANPR (Automatic Number Plate Recognition) with real-time tracking capabilities.

## Features

- **Vehicle Detection**: Detects cars, buses, trucks, and pedestrians using YOLO
- **Vehicle Counting**: Counts vehicles entering and exiting defined ROI areas
- **ANPR Support**: Automatic Number Plate Recognition for detected vehicles
- **Real-time Tracking**: Uses DeepSORT for object tracking across frames
- **API Integration**: Communicates with external APIs for camera data and analytics
- **Screenshot Capture**: Saves screenshots of detected events
- **Multi-camera Support**: Processes multiple camera feeds simultaneously
- **ROI Configuration**: Configurable Region of Interest for each camera

## Requirements

See `requirements.txt` for a complete list of dependencies. Main requirements include:

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- Deep SORT
- NumPy
- Requests

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd backup_vms
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your YOLO model files in the `models/` directory:
   - `best.pt` (vehicle detection model)
   - `ANPR.pt` (number plate recognition model)

## Configuration

### Environment Variables

- `API_IP`: IP address and port for the API server (default: 192.168.1.29:8001)

### Camera Configuration

Cameras are configured through the API endpoint. Each camera should have:
- Camera ID
- RTSP URL
- AI Models configuration
- Location information

### ROI Configuration

ROI (Region of Interest) points are saved automatically in `data/roi_configs/` as JSON files for each camera.

## Usage

Run the main application:

```bash
python main.py
```

The system will:
1. Fetch camera configurations from the API
2. Process video streams from configured cameras
3. Detect and track vehicles
4. Count entries/exits and perform ANPR
5. Send analytics data back to the API
6. Save screenshots of events

## Project Structure

```
backup_vms/
├── main.py              # Main application
├── requirements.txt     # Python dependencies
├── data/               # Data directory
│   ├── logs/           # Log files
│   ├── screenshots/    # Event screenshots
│   └── roi_configs/    # ROI configuration files
├── models/             # YOLO model files
├── Dataset/            # Sample frames/dataset
└── vms/               # Additional modules
```

## API Endpoints

- **Input**: `/api/v1/aiAnalytics/getCamerasForAnalytics` - Fetch camera configurations
- **Output**: `/api/v1/aiAnalytics/sendAnalyticsJson` - Send analytics data
- **Error**: `/api/v1/aiAnalytics/reportError` - Report errors

## Configuration Parameters

- `CROSSING_TOLERANCE`: Pixel tolerance for line crossing detection (default: 10)
- `FRAME_SKIP`: Process every Nth frame to reduce CPU usage (default: 5)
- `MAX_THREADS`: Maximum concurrent camera processing threads (default: 4)
- `SCREENSHOT_TIME_WINDOW`: Seconds to treat multiple detections as one (default: 5)

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
