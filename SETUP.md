# Event Tracking System - Setup Guide

Complete setup instructions for Windows, macOS, and Linux.

## 📋 Prerequisites

- Python 3.10+
- PostgreSQL 14+
- Node.js 18+
- Git
- Docker (optional)

## 🪟 Windows Setup

### 1. Install PostgreSQL

```powershell
# Using Chocolatey
choco install postgresql

# Or download from: https://www.postgresql.org/download/windows/
```

Create the database:
```powershell
# Open PostgreSQL command line or use pgAdmin
createdb -U postgres event_tracking
```

### 2. Install Python Dependencies

```powershell
cd C:\Users\<YourUser>\Documents\GitHub\ultralytics

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r examples/requirements.txt
```

### 3. Initialize Database

```powershell
cd examples
python -c "from db_models import init_db; init_db()"
```

### 4. Install Node.js Dependencies

```powershell
cd ..\frontend
npm install
```

## 🚀 Running the System

### Option A: Manual Start (3 Terminals)

**Terminal 1 - Event Tracking**
```powershell
cd C:\Users\<YourUser>\Documents\GitHub\ultralytics\examples
python integrated_tracker.py
```

**Terminal 2 - FastAPI Backend**
```powershell
cd C:\Users\<YourUser>\Documents\GitHub\ultralytics\examples
python api_backend.py
```

**Terminal 3 - Next.js Dashboard**
```powershell
cd C:\Users\<YourUser>\Documents\GitHub\ultralytics\frontend
npm run dev
```

### Option B: Using Docker Compose

```powershell
cd C:\Users\<YourUser>\Documents\GitHub\ultralytics
docker-compose up
```

## 🌐 Access Services

| Service | URL | Notes |
|---------|-----|-------|
| Dashboard | http://localhost:3000 | Next.js frontend |
| API | http://localhost:8000 | FastAPI backend |
| API Docs | http://localhost:8000/docs | Swagger UI |
| API ReDoc | http://localhost:8000/redoc | Alternative API docs |
| PostgreSQL | localhost:5432 | Database |

## 📝 Configuration

### Change Camera Source

Edit `integrated_tracker.py`:
```python
if __name__ == "__main__":
    service = EventService()
    tracker = IntegratedEventTracker(
        camera_source=0,  # Change: 0 = default webcam, "rtsp://..." = IP camera
        camera_id="cam_0"
    )
```

### Change Database URL

Edit `db_models.py`:
```python
DATABASE_URL = "postgresql://user:password@hostname:5432/database_name"
```

### Change API Port

Edit `api_backend.py`:
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Change port here
```

## 🧪 Testing

### Test Python Components

```powershell
# Test event service
python -c "
from examples.event_service import EventService, EventType, DetectionEvent
from datetime import datetime

service = EventService()
event = DetectionEvent(
    timestamp=datetime.now(),
    camera_id='test',
    track_id=1,
    event_type=EventType.ENTRY,
    class_name='person',
    confidence=0.95,
    x=100, y=150,
    x_min=50, y_min=100,
    x_max=150, y_max=200
)
service.record_event(event)
print(service.get_stats())
"

# Test API
python -m pytest tests/
```

### Test API Endpoints

```powershell
# Using PowerShell
$body = @{
    camera_id = 'cam_0'
    track_id = 1
    event_type = 'entry'
    class_name = 'person'
    confidence = 0.95
    x = 100
    y = 150
    x_min = 50
    y_min = 100
    x_max = 150
    y_max = 200
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/events" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body

# Get summary
Invoke-WebRequest -Uri "http://localhost:8000/summary" -Method GET
```

## 📧 Troubleshooting

### PostgreSQL Connection Error
```
Error: could not connect to server

Solution: Ensure PostgreSQL is running
Services -> PostgreSQL
Or: pg_ctl -D "C:\Program Files\PostgreSQL\15\data" start
```

### Port Already in Use
```
Error: Address already in use

Solution: Change port in the respective service
Or: Kill the process using the port
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process
```

### Python Module Not Found
```
Error: ModuleNotFoundError: No module named 'fastapi'

Solution: Install requirements
pip install -r examples/requirements.txt
```

### Frontend API Connection Error
```
Error: CORS policy

Solution: Update CORS_ORIGINS in api_backend.py
Or: Check API is running on correct port
```

## 🔄 Updating

```powershell
# Update Python dependencies
pip install -r examples/requirements.txt --upgrade

# Update Node dependencies
cd frontend
npm update
```

## 🐳 Docker Setup

```powershell
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📊 Data Export

### Export to CSV
```powershell
python -c "
from examples.event_service import EventService
from examples.report_generator import export_events_csv

service = EventService()
events = service.get_events(camera_id='cam_0')
export_events_csv(events, 'events.csv')
"
```

### Export to PDF
```powershell
python -c "
from examples.event_service import EventService
from examples.report_generator import export_pdf

service = EventService()
stats = service.get_stats()
export_pdf(stats, 'report.pdf')
"
```

## 🔐 Security Considerations

- Change PostgreSQL password
- Use environment variables for credentials
- Enable HTTPS in production
- Restrict API access with authentication
- Use firewall rules for database access

## 📚 Next Steps

1. Configure multiple cameras
2. Set up database backups
3. Configure email alerts
4. Add user authentication
5. Deploy to production environment

---

**Need help?** Check the [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation.
