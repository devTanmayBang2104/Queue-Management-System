# 🧠 QueueVision AI — Intelligent Queue Management System

An AI-powered, real-time queue management and analytics system using Computer Vision. Detects people via YOLOv8, tracks them with ByteTrack, analyzes queue behavior, predicts wait times, and displays everything on a premium React dashboard.

![Architecture](https://img.shields.io/badge/Architecture-FastAPI%20+%20React-blue)
![CV](https://img.shields.io/badge/CV-YOLOv8%20+%20ByteTrack-green)
![DB](https://img.shields.io/badge/Database-SQLite-orange)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Dashboard                       │
│  (Live Feed · Stats · Charts · Heatmap · Alerts)        │
└──────────────────────┬──────────────────────────────────┘
                       │ WebSocket + REST API
┌──────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend                        │
│  ┌────────────────────────────────────────────────────┐ │
│  │              CV Pipeline (Background Thread)        │ │
│  │                                                    │ │
│  │  YOLOv8 → ByteTrack → Queue Logic → Analytics     │ │
│  │            → Predictor → Alerts → Heatmap          │ │
│  └────────────────────────────────────────────────────┘ │
│                                                         │
│  REST: /api/analytics · /api/prediction · /api/alerts   │
│  WS:   /ws (live frame + data streaming)                │
└──────────────────────┬──────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │     SQLite      │
              │  (queue_analytics.db) │
              └─────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Person Detection** | YOLOv8 nano model, person-class only, GPU auto-detect |
| **Multi-Object Tracking** | ByteTrack with Kalman Filter, handles occlusions |
| **Queue Detection** | ROI-based + velocity analysis (stationary = in queue) |
| **Wait Time Calculation** | Per-person entry/exit, avg/max/individual times |
| **Prediction** | Linear Regression + Little's Law for wait estimation |
| **Behavior Analytics** | Abandonment detection, crowd spikes, arrival/service rates |
| **Heatmap** | Gaussian-accumulated density map with JET colormap |
| **Smart Alerts** | Configurable thresholds with cooldown deduplication |
| **Live Dashboard** | React + Chart.js, glassmorphism dark theme |
| **Video Upload** | Upload video files or use webcam |

---

## 📁 Project Structure

```
CV Project Theory/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── config.py             # Centralized configuration
│   ├── database.py           # SQLAlchemy + SQLite
│   ├── pipeline.py           # CV pipeline orchestrator
│   ├── requirements.txt      # Python dependencies
│   ├── models/
│   │   ├── detector.py       # YOLOv8 person detection
│   │   ├── tracker.py        # ByteTrack implementation
│   │   ├── queue_detector.py # Queue membership logic
│   │   └── predictor.py      # Wait time prediction (ML)
│   └── utils/
│       ├── analytics.py      # Behavior analytics
│       ├── heatmap.py        # Density heatmap generation
│       └── alerts.py         # Smart alert system
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx           # Main dashboard app
│       ├── index.css         # Design system (glassmorphism)
│       └── components/
│           ├── VideoFeed.jsx    # Live annotated video
│           ├── QueueStats.jsx   # Stat cards
│           ├── Charts.jsx       # Time-series charts
│           ├── HeatmapView.jsx  # Heatmap display
│           ├── AlertPanel.jsx   # Alert notifications
│           └── ConfigPanel.jsx  # Threshold configuration
├── utils/
│   └── generate_sample_video.py  # Test video generator
├── data/                     # Database + uploads
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **pip** and **npm**
- (Optional) NVIDIA GPU + CUDA for faster inference

### 1. Clone & Setup Backend

```bash
# Install Python dependencies
cd backend
pip install -r requirements.txt
```

> **Note:** First run will download the YOLOv8n model (~6MB) automatically.

### 2. Generate Sample Video (Optional)

```bash
cd ..
python utils/generate_sample_video.py
```

This creates `data/sample_queue.mp4` — a 60-second synthetic queue video.

### 3. Start Backend

```bash
# From project root
python -m backend.main
```

Backend starts at **http://localhost:8000**

### 4. Setup & Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard opens at **http://localhost:5173**

### 5. Use the Dashboard

1. Open **http://localhost:5173** in your browser
2. Enter video source:
   - Type `0` for webcam
   - Or paste the path to a video file (e.g., `data/sample_queue.mp4`)
   - Or click **Upload** to upload a video file
3. Click **▶ Start** to begin processing
4. Watch real-time analytics update on the dashboard!

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/status` | Pipeline running status |
| `POST` | `/api/start` | Start pipeline `{"video_source": "0"}` |
| `POST` | `/api/stop` | Stop pipeline |
| `GET` | `/api/analytics` | Current queue analytics |
| `GET` | `/api/prediction` | Wait time prediction |
| `GET` | `/api/alerts` | Alert history |
| `GET` | `/api/heatmap` | Heatmap image (base64) |
| `GET` | `/api/history` | Historical stats |
| `GET/POST` | `/api/config` | Get/update configuration |
| `POST` | `/api/upload` | Upload video file |
| `WS` | `/ws` | Live data stream |

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:5173

---

## ⚙️ Configuration

Thresholds are configurable via the dashboard UI or API:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_queue_length_alert` | 10 | Queue length to trigger alert |
| `max_wait_time_alert` | 300s | Max wait time before alert |
| `crowd_spike_threshold` | 15 | People count for crowd alert |
| `confidence_threshold` | 0.3 | YOLO detection confidence |
| `velocity_threshold` | 3.0 | px/frame below = stationary |

---

## 🧠 How It Works

### Detection Pipeline (per frame)
1. **YOLOv8** detects all persons → bounding boxes + confidence
2. **ByteTrack** assigns persistent IDs across frames using Kalman Filter + Hungarian algorithm
3. **Queue Detector** checks if person is inside ROI and moving slowly → in queue
4. **Analytics** computes wait times, abandonment, arrival/service rates
5. **Predictor** estimates wait time using Linear Regression + Little's Law
6. **Heatmap** accumulates positions with Gaussian splash + decay
7. **Alerts** check thresholds with cooldown deduplication
8. Frame is annotated with boxes/IDs and broadcast via WebSocket

### ByteTrack Two-Stage Association
1. Match **high-confidence** detections with existing tracks (IoU)
2. Match **low-confidence** detections with remaining unmatched tracks
3. This handles partial occlusions that single-threshold trackers miss

---

## 📊 Performance

| Metric | CPU (i7) | GPU (RTX 3060) |
|--------|----------|----------------|
| Detection | ~8 FPS | ~35 FPS |
| Full Pipeline | ~5-8 FPS | ~25-30 FPS |
| Model Size | 6.2 MB | 6.2 MB |

---

## 📄 License

MIT License — free for educational and commercial use.
