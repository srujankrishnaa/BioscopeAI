# BioscopeAI Deployment Guide

## Architecture
- **Frontend**: Deployed on Vercel (Static React App)
- **Backend**: Deployed on Railway (FastAPI Server)

## Backend Deployment (Railway)

### 1. Setup Railway Account
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Connect your repository

### 2. Deploy Backend
1. Create new project in Railway
2. Connect GitHub repository
3. Select `biomass-prediction-pixelwise/backend` as root directory
4. Railway will auto-detect Python and deploy

### 3. Environment Variables (Railway)
Set these in Railway dashboard:
```
PORT=8000
PYTHONPATH=/app
```

### 4. Get Backend URL
After deployment, Railway provides a URL like:
`https://your-app-name.railway.app`

## Frontend Deployment (Vercel)

### 1. Update Environment Variables
1. Copy your Railway backend URL
2. Update `frontend/.env.production`:
```
REACT_APP_API_URL=https://your-backend-url.railway.app
```

### 2. Deploy to Vercel
1. Go to [vercel.com](https://vercel.com)
2. Import GitHub repository
3. Set root directory to `biomass-prediction-pixelwise/frontend`
4. Deploy

### 3. Environment Variables (Vercel)
In Vercel dashboard, add:
```
REACT_APP_API_URL=https://your-backend-url.railway.app
REACT_APP_ENVIRONMENT=production
```

## Local Development
```bash
# Terminal 1: Backend
cd biomass-prediction-pixelwise/backend
uvicorn app.main:app --reload

# Terminal 2: Frontend  
cd biomass-prediction-pixelwise/frontend
npm run dev
```

## Production URLs
- **Frontend**: https://your-app.vercel.app
- **Backend**: https://your-backend.railway.app
- **API Docs**: https://your-backend.railway.app/docs

## Troubleshooting

### CORS Issues
If you get CORS errors, update `backend/app/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-app.vercel.app"],  # Add your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Environment Variables Not Loading
- Ensure `.env.production` exists in frontend
- Check Vercel dashboard environment variables
- Restart deployment after changes

## Cost
- **Railway**: Free tier (500 hours/month)
- **Vercel**: Free tier (unlimited for personal projects)
- **Total**: $0/month for small projects