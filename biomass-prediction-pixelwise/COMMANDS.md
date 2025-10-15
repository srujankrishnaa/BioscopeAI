# 🚀 Command Reference - Urban AGB Prediction System

## ⚡ Quick Start (3 Commands)

```bash
# 1. Install dependencies
cd backend && pip install -r requirements_minimal.txt

# 2. Start backend (Terminal 1)
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 3. Start frontend (Terminal 2)
cd ../frontend && npm start
```

Then open: **http://localhost:3000/urban-agb**

---

## 🧪 Testing Commands

```bash
# Run full test suite
cd backend && python test_prediction.py

# Test data fetcher only
python -m app.models.gee_data_fetcher

# Test API with curl
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"city":"Bangalore"}'

# Check API health
curl http://localhost:8000/health

# View API documentation
# Open: http://localhost:8000/docs
```

---

## 📦 Installation Commands

```bash
# Backend dependencies (minimal)
cd backend
pip install fastapi uvicorn numpy matplotlib pillow requests earthengine-api

# Or from file
pip install -r requirements_minimal.txt

# Frontend dependencies
cd frontend
npm install
```

---

## 🔍 Debugging Commands

```bash
# Check if backend is running
curl http://localhost:8000/health

# Check system status
curl http://localhost:8000/api/system-status

# View generated heatmaps
ls backend/outputs/heatmaps/

# View recent logs (while backend running)
# Check terminal where uvicorn is running

# Test specific city
cd backend
python -c "from app.models.gee_data_fetcher import quick_predict; print(quick_predict('Mumbai'))"
```

---

## 🛠️ Port Management

```bash
# If port 8000 is busy (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# If port 8000 is busy (Linux/Mac)
lsof -ti:8000 | xargs kill -9

# If port 3000 is busy
# Similar commands with :3000
```

---

## 📊 API Endpoints

```bash
# Health check
GET http://localhost:8000/health

# System status
GET http://localhost:8000/api/system-status

# Predict biomass
POST http://localhost:8000/api/predict
Body: {"city": "CityName"}

# Get supported cities
GET http://localhost:8000/api/cities

# View heatmap
GET http://localhost:8000/outputs/heatmaps/<filename>.png
```

---

## 🔄 Development Workflow

```bash
# 1. Make changes to backend code
# Backend auto-reloads (--reload flag)

# 2. Test changes
python test_prediction.py

# 3. View API docs
# Open: http://localhost:8000/docs

# 4. Test in frontend
# Open: http://localhost:3000/urban-agb
```

---

## 🎯 Common Tasks

### Test a specific city
```bash
cd backend
python -c "
from app.models.gee_data_fetcher import quick_predict
import json
result = quick_predict('Delhi')
print(json.dumps(result, indent=2))
"
```

### View all heatmaps
```bash
cd backend/outputs/heatmaps
ls -lht  # Linux/Mac
dir /o-d  # Windows
```

### Clean outputs
```bash
cd backend
rm -rf outputs/heatmaps/*  # Linux/Mac
del outputs\heatmaps\*  # Windows
```

### Check dependencies
```bash
cd backend
pip list | grep -E "fastapi|uvicorn|numpy|matplotlib"
```

---

## 📝 Example API Calls

### Using Python
```python
import requests

response = requests.post(
    "http://localhost:8000/api/predict",
    json={"city": "Bangalore"}
)

data = response.json()
print(f"AGB: {data['current_agb']['total_agb']:.2f} Mg/ha")
print(f"Heatmap: {data['heat_map']['image_url']}")
```

### Using JavaScript (fetch)
```javascript
fetch('http://localhost:8000/api/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({city: 'Mumbai'})
})
.then(r => r.json())
.then(data => console.log(data));
```

### Using curl with pretty print
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"city":"Chennai"}' | python -m json.tool
```

---

## 🚨 Troubleshooting

### Backend won't start
```bash
# Check Python version (need 3.8+)
python --version

# Check if port is available
netstat -ano | findstr :8000  # Windows
lsof -i:8000  # Linux/Mac

# Reinstall dependencies
pip install --force-reinstall -r requirements_minimal.txt
```

### Frontend won't connect
```bash
# Check backend is running
curl http://localhost:8000/health

# Check CORS in backend logs
# Should see: "CORS middleware enabled"

# Clear npm cache
cd frontend
rm -rf node_modules
npm install
```

### No predictions working
```bash
# Test data fetcher directly
cd backend
python -m app.models.gee_data_fetcher

# If that works, test API
curl http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"city":"Mumbai"}'

# Check logs in terminal
```

---

## 📂 Directory Structure

```
biomass-prediction-pixelwise/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── prediction.py          ← API endpoints
│   │   ├── models/
│   │   │   └── gee_data_fetcher.py    ← Data fetching
│   │   └── main.py                     ← FastAPI app
│   ├── outputs/
│   │   └── heatmaps/                   ← Generated images
│   ├── requirements_minimal.txt
│   └── test_prediction.py              ← Test suite
├── frontend/
│   ├── src/
│   │   ├── services/
│   │   │   └── urbanAgbService.ts     ← API client
│   │   └── pages/
│   │       └── UrbanAGBPage.tsx       ← Main UI
│   └── package.json
├── QUICK_START_GUIDE.md               ← Start here!
├── IMPLEMENTATION_SUMMARY.md          ← What was built
└── COMMANDS.md                        ← This file
```

---

## 🎓 Learning Resources

```bash
# View API documentation
http://localhost:8000/docs

# Read implementation details
cat IMPLEMENTATION_SUMMARY.md

# Check quick start guide
cat QUICK_START_GUIDE.md

# Explore code
code backend/app/api/prediction.py
code backend/app/models/gee_data_fetcher.py
```

---

## ⚡ One-Liner Test

```bash
# Complete test in one command
cd backend && python -c "from app.models.gee_data_fetcher import quick_predict; r=quick_predict('Bangalore'); print(f'✅ AGB: {r[\"current_agb\"][\"total_agb\"]:.1f} Mg/ha, Canopy: {r[\"current_agb\"][\"canopy_cover\"]:.1f}%')"
```

---

## 🎯 Success Indicators

When everything is working:

1. ✅ Backend: `INFO: Uvicorn running on http://0.0.0.0:8000`
2. ✅ Health: `curl localhost:8000/health` returns `{"status":"healthy"}`
3. ✅ Frontend: Browser shows Urban AGB page
4. ✅ Prediction: Returns results in <30 seconds
5. ✅ Heatmap: New PNG file in `outputs/heatmaps/`

---

**Quick help: If stuck, run `python backend/test_prediction.py`** 🚀

