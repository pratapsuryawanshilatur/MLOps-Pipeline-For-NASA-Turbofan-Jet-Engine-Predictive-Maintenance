# MLOps Pipeline for NASA Turbofan Predictive Maintenance

A complete MLOps pipeline for predictive maintenance of jet engines using NASA's Turbofan dataset.

## Features

- **Data Pipeline**: Apache Airflow for data ingestion and preprocessing
- **Model Training**: Dockerized Random Forest with hyperparameter tuning
- **CI/CD**: GitHub Actions for automated testing
- **MLOps Best Practices**: Reproducible, versioned, automated workflows

## Project Structure
mlops-turbofan-project/
├── .github/workflows/ # CI/CD pipelines
├── dags/ # Airflow workflows
├── scripts/ # Python scripts
├── data/ # Datasets
├── models/ # Trained models (git-ignored)
└── docker-compose.yml # Container orchestration


## Quick Start

1. **Clone repository**
2. **Setup**: `docker-compose up -d`
3. **Access Airflow**: http://localhost:8081 (admin/admin)
4. **Run pipelines**: Trigger DAGs in Airflow UI

## CI/CD Pipeline

Automated tests run on every push:
- Unit tests
- Data validation
- Model training on sample data
- Quality metric checks

## Technologies

- Apache Airflow
- Docker
- Scikit-learn
- GitHub Actions
<<<<<<< HEAD
- Pandas, NumPy
=======
- Pandas, NumPy
>>>>>>> 4a01d67 (Add GitOps structure for ArgoCD)
