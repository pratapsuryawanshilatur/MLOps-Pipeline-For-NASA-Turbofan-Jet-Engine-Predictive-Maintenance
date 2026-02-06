NASA Turbofan Predictive Maintenance - End-to-End MLOps Pipeline

Project Overview
A complete MLOps pipeline for NASA's Turbofan Engine Degradation Dataset that demonstrates the full machine learning lifecycle from data ingestion to production deployment with GitOps automation. This project implements predictive maintenance for jet engines using a Random Forest classifier deployed with modern DevOps practices.

Key Features:
📊 Automated Data Pipeline: Apache Airflow DAGs for data ingestion and preprocessing
🤖 Machine Learning: Random Forest Classifier with 96.6% accuracy
🚀 Production API: FastAPI service with health checks and batch predictions
🐳 Containerized: Docker containers for all components
⚓ Kubernetes: Multi-replica deployment with PersistentVolumes
🔄 GitOps: ArgoCD for automated deployment from Git
🔧 CI/CD: GitHub Actions for automated testing and validation

Model Performance:

Metric	    Score
Accuracy	96.63%
F1 Score	0.8863
ROC AUC	    0.9906
Precision	0.9048
Recall	    0.8684

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                     MLOps Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │  Kaggle │  │  Airflow │  │  ML      │  │  FastAPI   │  │
│  │  Data   │──│  Pipeline│──│  Training│──│  Serving   │  │
│  └─────────┘  └──────────┘  └──────────┘  └────────────┘  │
│         │            │            │               │        │
│         └────────────┴────────────┴───────────────┼────────┘
│                                                   │
│  ┌────────────────────────────────────────────────┼────────┐
│  │           Kubernetes + ArgoCD Deployment       │        │
│  │  ┌────────────┐  ┌──────────┐  ┌──────────┐   │        │
│  │  │ Deployment │  │ Service  │  │  Ingress │◀──┘        │
│  │  │  (2 Pods)  │  │ (8000)   │  │          │            │
│  │  └────────────┘  └──────────┘  └──────────┘            │
│  │                                                        │
│  │  ┌────────────┐  ┌──────────┐                        │
│  │  │ ConfigMap  │  │   PVC    │                        │
│  │  │  (Env Vars)│  │ (Models) │                        │
│  │  └────────────┘  └──────────┘                        │
│  └───────────────────────────────────────────────────────┘

Project Structure:
mlops-turbofan-project/
├── .github/workflows/
│   └── mlops-ci.yml              # GitHub Actions CI/CD pipeline
├── dags/
│   ├── turbofan_data_pipeline.py     # Airflow: Data processing DAG
│   └── turbofan_model_training_dag.py # Airflow: Model training DAG
├── scripts/
│   ├── data_processor.py         # Data preprocessing logic
│   ├── train_model.py            # Model training logic
│   ├── app.py                    # FastAPI application
│   ├── test_api.py               # API test client
│   └── requirements.txt          # Python dependencies
├── k8s/
│   ├── base/                     # Common Kubernetes manifests
│   │   ├── configmap.yaml
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── persistent-volume.yaml
│   │   └── persistent-volume-claim.yaml
│   ├── overlays/
│   │   └── staging/              # Staging environment configurations
│   └── argocd/
│       └── turbofan-app.yaml     # ArgoCD Application manifest
├── data/
│   ├── raw/                      # NASA original data
│   └── processed/                # Cleaned data (git-ignored)
├── models/                       # Trained models (git-ignored)
├── Dockerfile                    # Training container
├── Dockerfile.airflow            # Custom Airflow with ML packages
├── Dockerfile.api                # FastAPI container
├── docker-compose.yml            # Multi-service orchestration
└── README.md                     # This file

Quick Start:

Prerequisites
Docker & Docker Compose
Kubernetes (Docker Desktop or Minikube)
Python 3.9+
Git

1. Local Development Setup:
# Clone the repository
git clone https://github.com/pratapsuryawanshilatur/MLOps-Pipeline-For-NASA-Turbofan-Jet-Engine-Predictive-Maintenance.git
cd MLOps-Pipeline-For-NASA-Turbofan-Jet-Engine-Predictive-Maintenance

# Start all services with Docker Compose
docker-compose up -d

# Access services:
# Airflow UI: http://localhost:8081
# FastAPI: http://localhost:8000
# API Docs: http://localhost:8000/docs

2. Kubernetes Deployment with ArgoCD:
# Install ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Expose ArgoCD UI
kubectl patch svc argocd-server -n argocd -p '{"spec": {"type": "NodePort"}}'

# Get ArgoCD credentials
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d

# Deploy the application via ArgoCD
kubectl apply -f k8s/argocd/turbofan-app.yaml

Example API Usage:
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 0.3, 0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3]}'

# Model information
curl http://localhost:8000/model/info

Data Pipeline:
1. The data pipeline processes the NASA Turbofan Engine Degradation Dataset (FD001):
Data Ingestion: Raw CSV data from /data/raw/
2. Preprocessing:
Calculate Remaining Useful Life (RUL)
Create binary classification target (Failure/No Failure)
Handle missing values and normalize features
3. Storage: Save processed data as Parquet files in /data/processed/

Machine Learning Model:
Algorithm: Random Forest Classifier
Hyperparameter Tuning: GridSearchCV with 5-fold cross-validation

Docker Services:

Service	           Image	            Port	Description
PostgreSQL	       postgres:13	        5432	Airflow metadata database
Airflow Webserver	Custom	            8081	Airflow UI
Airflow Scheduler	Custom	             -	    Airflow scheduler
FastAPI	           turbofan-api:v1.0.2	8000	Model serving API

Kubernetes Resources:

Resource	          Type	                Purpose
turbofan-api	      Deployment	        Runs 2 replicas of FastAPI
turbofan-api-service  Service	            Exposes API internally
turbofan-api-config	  ConfigMap	            Environment variables
turbofan-models-pv	 PersistentVolume    	Model storage
turbofan-models-pvc	 PersistentVolumeClaim	PVC for model storage

GitOps Workflow:
Code Push: Changes pushed to GitHub
CI Pipeline: GitHub Actions runs tests
ArgoCD Detection: ArgoCD detects changes in Git
Auto-Sync: ArgoCD synchronizes cluster state with Git
Self-Healing: ArgoCD continuously reconciles state

Testing:
# Run all tests
python -m pytest scripts/

# Test API locally
python scripts/test_api.py

# CI/CD pipeline automatically runs on every push

Monitoring & Logging:
Airflow UI: Monitor DAG executions at http://localhost:8081
FastAPI Logs: Container logs show API requests and model predictions
Kubernetes: kubectl logs for pod-level monitoring
ArgoCD UI: Deployment synchronization status

Troubleshooting:
Common Issues--
1. Port conflicts: Ensure ports 8081, 8000, and 5432 are available
2. Docker resource limits: Increase Docker Desktop memory allocation
3. Kubernetes PVC pending: Check PV is created and bound
4. ArgoCD sync failures: Verify kustomization.yaml syntax

Debug Commands:
# Check all services
docker-compose ps

# View logs
docker-compose logs turbofan-api

# Kubernetes status
kubectl get all -n turbofan-staging

# ArgoCD status
kubectl get application -n argocd

Development:
Adding New Features
1. Data Pipeline: Modify scripts/data_processor.py
2. ML Model: Update scripts/train_model.py
3. API Endpoints: Edit scripts/app.py
4. K8s Manifests: Update files in k8s/ directory

Environment Variables:
Variable	Default 	                    Description
MODEL_PATH	models/turbofan_model_v1.joblib	Model file path
APP_ENV	    staging	                        Application environment
LOG_LEVEL	INFO	                        Logging level

Contributing:
Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit changes (git commit -m 'Add AmazingFeature')
Push to branch (git push origin feature/AmazingFeature)
Open a Pull Request

Contact:
Pratap Suryawanshi - pratapsuryawanshi98@gmail.com