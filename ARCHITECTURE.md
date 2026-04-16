# Event Tracking System - Complete Architecture

Complete YOLO-based event tracking system with real-time monitoring, API backend, and dashboard.

## 📐 Architecture

```
Caméras (RTSP / USB)
         ↓
YOLOv8 + Tracking
         ↓
Event Service (Python)
         ↓
API Backend (FastAPI)
         ↓
PostgreSQL (données)
         ↓
Frontend Dashboard (Next.js)
         ↓
Reports (PDF / CSV export)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Python dependencies
pip install fastapi uvicorn sqlalchemy psycopg2-binary reportlab

# Frontend dependencies
cd frontend
npm install
```

### 2. Database Setup (PostgreSQL)

```bash
# Create database
createdb event_tracking

# Initialize schema
python -c "from db_models import init_db; init_db()"
```

### 3. Start Services

**Terminal 1 - Event Tracking (Cameras + Event Service)**
```bash
python integrated_tracker.py
```

**Terminal 2 - FastAPI Backend**
```bash
python api_backend.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

**Terminal 3 - Next.js Dashboard**
```bash
cd frontend
npm run dev
# Dashboard at http://localhost:3000
```

## 📁 File Structure

```
examples/
├── event_tracking.py          # Main tracking script (cameras → tracking)
├── region_utils.py            # Region counting utilities
├── event_service.py           # Event recording and statistics
├── api_backend.py             # FastAPI REST API
├── db_models.py               # SQLAlchemy database models
├── report_generator.py        # PDF/CSV export utilities
├── integrated_tracker.py      # Combined tracking + event service
└── requirements.txt

frontend/
├── pages/
│   ├── index.tsx              # Dashboard home
│   ├── events.tsx             # Events list
│   └── reports.tsx            # Reports download
├── package.json
├── next.config.js
└── tailwind.config.js
```

## 🔌 API Endpoints

### Health Check
```
GET /health
→ {"status": "healthy"}
```

### Create Event
```
POST /events
Body: {
  "camera_id": "cam_0",
  "track_id": 1,
  "event_type": "entry",
  "class_name": "person",
  "confidence": 0.95,
  "x": 100, "y": 150,
  "x_min": 50, "y_min": 100,
  "x_max": 150, "y_max": 200,
  "region_name": "entrance"
}
```

### Get Events
```
GET /events?camera_id=cam_0&event_type=entry&limit=100
→ [{...}, {...}]
```

### Get Statistics
```
GET /stats
GET /stats/cam_0
→ {
  "camera_id": "cam_0",
  "entry_count": 42,
  "exit_count": 38,
  "current_crowd": 4,
  "unique_ids_count": 120,
  ...
}
```

### Get Summary
```
GET /summary
→ {
  "total_events": 1200,
  "total_cameras": 2,
  "total_entries": 500,
  "total_exits": 480,
  "total_crowd": 20,
  "cameras": {...}
}
```

### Export Reports
```
GET /export/csv?camera_id=cam_0
GET /export/pdf?camera_id=cam_0
```

## 🎯 Features

✅ **Real-time Tracking**
- Multiple camera support (USB, RTSP)
- YOLOv8 object detection
- ByteTrack object tracking
- Multi-threaded processing

✅ **Event Recording**
- Entry/Exit detection
- Region-based counting
- Custom geometry regions
- Event storage and retrieval

✅ **REST API**
- FastAPI with async/await
- Full CRUD operations
- Filtering and pagination
- Real-time statistics

✅ **Database**
- PostgreSQL for persistence
- Event history
- Camera sessions
- Region statistics
- Indexed queries

✅ **Dashboard**
- Real-time metrics
- Live camera statistics
- Event timeline
- Export functionality

✅ **Reports**
- CSV export
- PDF reports
- Data visualization
- Customizable summaries

## 📊 Usage Examples

### Python - Track with Event Service
```python
from event_service import EventService
from integrated_tracker import IntegratedEventTracker

service = EventService()
tracker = IntegratedEventTracker(0, camera_id="cam_0")
tracker.set_event_service(service)
tracker.run()

# Get statistics
stats = service.get_stats("cam_0")
print(stats)
```

### Python - Export Reports
```python
from report_generator import export_events_csv, export_pdf

events = service.get_events("cam_0")
export_events_csv(events, "events.csv")

stats = service.get_stats()
export_pdf(stats, "report.pdf")
```

### Python - Access API
```python
import requests

# Create event
response = requests.post(
    "http://localhost:8000/events",
    json={
        "camera_id": "cam_0",
        "track_id": 1,
        "event_type": "entry",
        "class_name": "person",
        "confidence": 0.95,
        "x": 100, "y": 150,
        "x_min": 50, "y_min": 100,
        "x_max": 150, "y_max": 200,
    }
)

# Get summary
summary = requests.get("http://localhost:8000/summary").json()
print(summary)
```

### JavaScript - Frontend Integration
```typescript
// Fetch stats from API
const response = await fetch('http://localhost:8000/summary');
const summary = await response.json();

// Update dashboard
setSummary(summary);
```

## 🔧 Configuration

### Event Service
```python
service = EventService()
service.max_events = 10000  # Keep last N events
```

### Tracker
```python
tracker = IntegratedEventTracker(
    camera_source=0,
    weights="yolov8n.pt",
    camera_id="cam_0"
)
```

### Database
```python
db_url = get_db_url(
    user="postgres",
    password="password",
    host="localhost",
    port=5432,
    db="event_tracking"
)
init_db(db_url)
```

## 📈 Scaling Considerations

- Multiple cameras: Use threading/asyncio
- High event volume: Add message queue (Redis, RabbitMQ)
- Real-time updates: WebSocket for dashboard
- Load balancing: Use Nginx reverse proxy
- Database: Consider partitioning by timestamp

## 🛠️ Development

```bash
# Code formatting
black examples/ frontend/

# Type checking
mypy examples/

# Linting
pylint examples/
eslint frontend/

# Testing
pytest tests/
npm test --prefix frontend/
```

## 📝 License

AGPL-3.0 - Ultralytics

## 🤝 Contributing

Contributions welcome! Please follow the style guide and add tests.

---

**Questions?** Check the [Ultralytics Docs](https://docs.ultralytics.com/)
