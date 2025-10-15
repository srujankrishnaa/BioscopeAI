# Above Ground Biomass Prediction System

A comprehensive system for predicting above ground biomass using satellite imagery, climate data, and machine learning models.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project provides an end-to-end solution for predicting above ground biomass (AGB) using a combination of satellite imagery, climate data, and machine learning models. The system includes:

- Data preprocessing pipeline for satellite and climate data
- CNN+LSTM model architecture for spatial-temporal prediction
- Model optimization and quantization for efficient deployment
- Interactive web interface for visualization and exploration
- RESTful API for programmatic access

## Architecture

The system is composed of the following main components:

### Backend
- **FastAPI**: Web framework for building the API
- **TensorFlow**: Machine learning framework for model training and inference
- **Xarray**: Data handling for multidimensional arrays
- **PostgreSQL**: Database for storing user data and predictions
- **Redis**: Caching layer for improved performance

### Frontend
- **React**: JavaScript library for building user interfaces
- **TypeScript**: Typed superset of JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **Folium**: Interactive map visualization

### ML Pipeline
- **Data Processor**: Handles satellite data preprocessing, alignment, and normalization
- **Model Builder**: Constructs and trains the CNN+LSTM model
- **Model Optimizer**: Performs hyperparameter tuning and model quantization
- **Predictor**: Handles inference and generates heatmaps and reports

## Features

### Data Processing
- Support for multiple satellite data sources (MODIS, SRTM, SMAP)
- Automatic data alignment and normalization
- Quality control and filtering
- Time series sequence generation

### Machine Learning
- CNN+LSTM architecture for spatial-temporal modeling
- Hyperparameter tuning with Keras Tuner
- Model quantization for efficient deployment
- Support for both Keras and TensorFlow Lite models

### Visualization
- Interactive biomass prediction heatmaps
- Time series visualization
- Spatial comparison tools
- PDF report generation
- Summary statistics and metrics

### API
- RESTful API for all functionality
- Authentication and authorization
- Rate limiting and caching
- OpenAPI documentation

## Installation

### Prerequisites

- Python 3.9+
- Node.js 18+
- Docker and Docker Compose
- PostgreSQL 13+
- Redis 6+

### Quick Start with Docker

1. Clone the repository:
```bash
git clone https://github.com/yourusername/biomass-prediction.git
cd biomass-prediction
```

2. Create a .env file based on the example:
```bash
cp backend/.env.example backend/.env
```

3. Build and run with Docker Compose:
```bash
docker-compose up --build
```

4. Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Manual Installation

#### Backend

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database:
```bash
python -m alembic upgrade head
```

6. Run the development server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

## Usage

### Data Preprocessing

To preprocess data for training:

```bash
cd backend
python -m app.models.data_processor
```

This will:
- Load satellite data from the dataset directory
- Apply quality control and alignment
- Generate normalized sequences
- Save processed data to the data/processed directory

### Model Training

To train a model:

```bash
cd backend
python -m app.models.model_builder
```

This will:
- Load preprocessed data
- Build and train the CNN+LSTM model
- Save the trained model to the models directory
- Generate training history and evaluation plots

### Model Optimization

To optimize a trained model:

```bash
cd backend
python -m app.models.model_optimizer
```

This will:
- Perform hyperparameter tuning
- Quantize the model for efficient deployment
- Generate optimization reports and comparisons

### Inference

To make predictions:

```bash
cd backend
python -m app.models.predictor
```

This will:
- Load a trained model and preprocessing parameters
- Fetch data for specified cities
- Generate biomass predictions
- Create heatmaps and reports

### API Usage

The API provides endpoints for all functionality. For example:

```python
import requests

# Get prediction for a city
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={
        "city_name": "Mumbai",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    }
)

prediction = response.json()
```

## API Documentation

The API documentation is automatically generated and available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Key Endpoints

#### Authentication
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/register` - User registration

#### Predictions
- `POST /api/v1/predict` - Create a new prediction
- `GET /api/v1/predictions/{id}` - Get a prediction by ID
- `GET /api/v1/predictions` - List all predictions

#### Data
- `POST /api/v1/data/preprocess` - Preprocess data
- `GET /api/v1/data/status` - Check data processing status

#### Models
- `POST /api/v1/models/train` - Train a model
- `GET /api/v1/models/{id}` - Get model information
- `POST /api/v1/models/optimize` - Optimize a model

## Project Structure

```
biomass-prediction/
├── backend/                 # Backend application
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI application entry point
│   │   ├── api/             # API endpoints
│   │   │   ├── __init__.py
│   │   │   ├── prediction.py
│   │   │   └── ...
│   │   ├── models/          # Machine learning models
│   │   │   ├── __init__.py
│   │   │   ├── data_processor.py
│   │   │   ├── model_builder.py
│   │   │   ├── model_optimizer.py
│   │   │   └── predictor.py
│   │   └── utils/           # Utility functions
│   │       ├── __init__.py
│   │       └── config.py
│   ├── models/             # Saved model files
│   ├── data/                # Data files
│   ├── outputs/             # Generated outputs
│   ├── requirements.txt    # Python dependencies
│   ├── .env                 # Environment variables
│   └── Dockerfile           # Docker configuration
├── frontend/               # Frontend application
│   ├── public/             # Static assets
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── services/       # API services
│   │   ├── types/          # TypeScript types
│   │   └── utils/          # Utility functions
│   ├── package.json        # Node.js dependencies
│   ├── tailwind.config.js  # Tailwind CSS configuration
│   ├── tsconfig.json       # TypeScript configuration
│   ├── nginx.conf          # Nginx configuration
│   └── Dockerfile          # Docker configuration
├── dataset/                # Sample datasets
├── docker-compose.yml      # Docker Compose configuration
└── README.md              # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Use TypeScript for frontend development
- Write tests for new features
- Update documentation as needed
- Use conventional commit messages

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NASA Earthdata for satellite imagery
- GEDI team for biomass data
- TensorFlow and Keras teams for ML frameworks
- FastAPI and React communities