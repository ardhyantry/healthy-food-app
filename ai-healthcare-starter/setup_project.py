#!/usr/bin/env python3
"""
Setup script to create the complete project structure for AI Healthcare Diagnostics.
Run this script to automatically generate all necessary directories and boilerplate files.
"""

import os
from pathlib import Path
import shutil


def create_directory_structure():
    """Create the complete directory structure for the project."""
    
    directories = [
        # Main source directories
        "src",
        "src/config",
        "src/data", 
        "src/models",
        "src/training",
        "src/evaluation",
        "src/explainability",
        "src/api",
        "src/monitoring",
        "src/utils",
        
        # Test directories
        "tests",
        "tests/unit",
        "tests/integration", 
        "tests/performance",
        
        # Notebooks
        "notebooks",
        
        # Configuration
        "configs",
        
        # Scripts
        "scripts",
        
        # Docker
        "docker",
        "docker/k8s",
        
        # Data directories
        "data",
        "data/raw",
        "data/processed",
        "data/external",
        
        # Model artifacts
        "models",
        "models/trained_models",
        "models/model_artifacts",
        
        # Results and reports
        "results",
        "results/experiments",
        "results/evaluations",
        "results/reports",
        
        # Documentation
        "docs",
        "docs/images",
        
        # GitHub workflows
        ".github",
        ".github/workflows",
        
        # MLflow artifacts
        "artifacts",
        "mlruns",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if directory.startswith("src/") or directory == "src" or directory.startswith("tests/"):
            init_file = Path(directory) / "__init__.py"
            if not init_file.exists():
                init_file.touch()
    
    print("✅ Directory structure created successfully!")


def create_boilerplate_files():
    """Create essential boilerplate files."""
    
    files_content = {
        # Configuration files
        "src/config/config.py": '''"""Configuration management for the AI Healthcare Diagnostics project."""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Project settings
    PROJECT_NAME: str = "AI Healthcare Diagnostics"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Data settings
    DATA_DIR: Path = Path("data")
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    
    # Model settings
    MODEL_DIR: Path = Path("models")
    TRAINED_MODELS_DIR: Path = MODEL_DIR / "trained_models"
    
    # Training settings
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    EPOCHS: int = 100
    NUM_CLASSES: int = 7
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # MLflow settings
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "skin-cancer-detection"
    
    # Database settings
    DATABASE_URL: Optional[str] = None
    REDIS_URL: str = "redis://localhost:6379"
    
    class Config:
        env_file = ".env"


settings = Settings()
''',

        "src/config/logging_config.py": '''"""Logging configuration for the project."""

import logging
import logging.config
from pathlib import Path


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "logs/app.log",
        },
    },
    "loggers": {
        "": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
}


def setup_logging():
    """Set up logging configuration."""
    Path("logs").mkdir(exist_ok=True)
    logging.config.dictConfig(LOGGING_CONFIG)
''',

        # Main model file
        "src/models/skin_cancer_classifier.py": '''"""Skin cancer classification model implementation."""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from typing import Tuple, Optional


class SkinCancerClassifier(nn.Module):
    """EfficientNet-based skin cancer classifier with uncertainty quantification."""
    
    def __init__(
        self, 
        num_classes: int = 7, 
        model_name: str = 'efficientnet-b4',
        uncertainty: bool = True,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.uncertainty = uncertainty
        
        # Load pre-trained EfficientNet
        self.backbone = EfficientNet.from_pretrained(model_name)
        self.backbone._fc = nn.Identity()
        
        # Get feature dimension
        feature_dim = self.backbone._fc.in_features
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(512, num_classes)
        )
        
        # Uncertainty estimation head
        if uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the network."""
        features = self.backbone(x)
        predictions = self.classifier(features)
        
        uncertainty = None
        if self.uncertainty:
            uncertainty = self.uncertainty_head(features)
        
        return predictions, uncertainty
    
    @classmethod
    def load_pretrained(cls, model_path: str, **kwargs) -> 'SkinCancerClassifier':
        """Load a pre-trained model."""
        model = cls(**kwargs)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    
    def save_model(self, model_path: str):
        """Save the model."""
        torch.save(self.state_dict(), model_path)
''',

        # API main file
        "src/api/main.py": '''"""FastAPI main application for skin cancer detection."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, Any
import logging

from src.config.config import settings
from src.config.logging_config import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-powered skin cancer detection API"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {"message": "AI Healthcare Diagnostics API", "version": settings.VERSION}


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/v1/predict")
async def predict_skin_lesion(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Predict skin lesion type from uploaded image."""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # TODO: Implement prediction logic
        # For now, return a mock response
        mock_response = {
            "predictions": {
                "Melanoma": 0.15,
                "Melanocytic nevus": 0.65,
                "Basal cell carcinoma": 0.10,
                "Actinic keratosis": 0.05,
                "Benign keratosis": 0.03,
                "Dermatofibroma": 0.01,
                "Vascular lesion": 0.01
            },
            "confidence": 0.87,
            "recommendation": "Consult with a dermatologist for professional evaluation."
        }
        
        logger.info(f"Processed image: {file.filename}")
        return mock_response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
''',

        # Docker files
        "docker/Dockerfile.api": '''FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
''',

        "docker/docker-compose.yml": '''version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - DEBUG=false
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    volumes:
      - ../models:/app/models
      - ../data:/app/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  redis_data:
  grafana_data:
''',

        # Environment template
        ".env.example": '''# AI Healthcare Diagnostics Configuration

# Application settings
DEBUG=true
PROJECT_NAME="AI Healthcare Diagnostics"
VERSION="1.0.0"

# API settings
API_HOST=0.0.0.0
API_PORT=8000

# MLflow settings
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=skin-cancer-detection

# Database settings
DATABASE_URL=postgresql://user:password@localhost/ai_healthcare
REDIS_URL=redis://localhost:6379

# AWS settings (for cloud deployment)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
AWS_BUCKET_NAME=your_bucket_name

# Model settings
BATCH_SIZE=32
LEARNING_RATE=0.0001
EPOCHS=100
NUM_CLASSES=7
''',

        # GitHub Actions CI/CD
        ".github/workflows/ci.yml": '''name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Lint with flake8
      run: |
        flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src tests --count --max-complexity=10 --max-line-length=88 --statistics
    
    - name: Test with pytest
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
''',

        # Gitignore
        ".gitignore": '''# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Project specific
data/raw/
data/processed/
models/trained_models/
models/model_artifacts/
mlruns/
artifacts/
logs/
*.log

# MacOS
.DS_Store

# Windows
Thumbs.db
ehthumbs.db
Desktop.ini
''',
    }
    
    for file_path, content in files_content.items():
        file_path_obj = Path(file_path)
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path_obj, 'w') as f:
            f.write(content)
    
    print("✅ Boilerplate files created successfully!")


def create_notebooks():
    """Create starter Jupyter notebooks."""
    
    notebooks_content = {
        "notebooks/01_exploratory_data_analysis.ipynb": {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# Exploratory Data Analysis\\n", "\\n", "This notebook contains the exploratory data analysis for the skin cancer detection dataset."]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": ["import pandas as pd\\n", "import numpy as np\\n", "import matplotlib.pyplot as plt\\n", "import seaborn as sns\\n", "from pathlib import Path\\n", "\\n", "# Set up plotting\\n", "plt.style.use('seaborn-v0_8')\\n", "sns.set_palette('husl')"]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
    }
    
    import json
    
    for notebook_path, notebook_content in notebooks_content.items():
        Path(notebook_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
    
    print("✅ Starter notebooks created successfully!")


def main():
    """Main setup function."""
    print("🚀 Setting up AI Healthcare Diagnostics project structure...")
    print()
    
    create_directory_structure()
    create_boilerplate_files()
    create_notebooks()
    
    print()
    print("🎉 Project setup complete!")
    print()
    print("Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Copy .env.example to .env and configure your settings")
    print("3. Start with the implementation roadmap in IMPLEMENTATION_ROADMAP.md")
    print("4. Begin with Week 1 tasks: data acquisition and preprocessing")
    print()
    print("Happy coding! 🔬🤖")


if __name__ == "__main__":
    main()
''',
    }
    
    for file_path, content in files_content.items():
        file_path_obj = Path(file_path)
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path_obj, 'w') as f:
            f.write(content)
    
    print("✅ Boilerplate files created successfully!")


def create_notebooks():
    """Create starter Jupyter notebooks."""
    
    notebooks_content = {
        "notebooks/01_exploratory_data_analysis.ipynb": {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# Exploratory Data Analysis\\n", "\\n", "This notebook contains the exploratory data analysis for the skin cancer detection dataset."]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": ["import pandas as pd\\n", "import numpy as np\\n", "import matplotlib.pyplot as plt\\n", "import seaborn as sns\\n", "from pathlib import Path\\n", "\\n", "# Set up plotting\\n", "plt.style.use('seaborn-v0_8')\\n", "sns.set_palette('husl')"]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
    }
    
    import json
    
    for notebook_path, notebook_content in notebooks_content.items():
        Path(notebook_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
    
    print("✅ Starter notebooks created successfully!")


def main():
    """Main setup function."""
    print("🚀 Setting up AI Healthcare Diagnostics project structure...")
    print()
    
    create_directory_structure()
    create_boilerplate_files()
    create_notebooks()
    
    print()
    print("🎉 Project setup complete!")
    print()
    print("Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Copy .env.example to .env and configure your settings")
    print("3. Start with the implementation roadmap in IMPLEMENTATION_ROADMAP.md")
    print("4. Begin with Week 1 tasks: data acquisition and preprocessing")
    print()
    print("Happy coding! 🔬🤖")


if __name__ == "__main__":
    main()