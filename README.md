# Intelligent Claims Triage System

An enterprise-grade Machine Learning system designed for decision orchestration, human-in-the-loop review, and explainability in insurance workflows. Instead of just exposing raw predictions, this system implements confidence-based triage policies to dynamically route insurance claims. High-confidence decisions are auto-processed, while marginal cases are queued for human review with explainable AI (SHAP) overlays, capturing audit trails of human overrides to continuously measure and monitor decision quality and data drift.

---

## System Architecture

```
                       [ Incoming Claims ]
                                |
                                v
                      [ Data Validation ]
                                |
                                v
                    [ Feature Engineering ]
                                |
                                v
                 [ AWS SageMaker Inference ]
                                |
                      (Probability Score)
                                |
             +------------------+------------------+
             |                                     |
    [ Prob >= 0.90 ]                     [ 0.55 <= Prob < 0.90 ]
             |                                     |
             v                                     v
     [ Auto-Reject ]                       [ Human Review Queue ]
             |                                     |
             v                                     v
    [ Audit Database ] <=== (Override/Notes) === [ Streamlit App ]
                                                   (SHAP Explainability)
```

---

## Tech Stack

| Component | Technology | Role |
|---|---|---|
| **Core Language** | Python 3.11 | System backend, pipelines, and ML logic |
| **Backend API** | FastAPI + SQLModel | Serves prediction endpoint and manages claims state/CRUD |
| **Model Hosting** | AWS SageMaker | Real-time endpoint hosting and model registry management |
| **Storage** | SQLite / AWS S3 | Local database for audit trails; S3 for training dataset artifacts |
| **Explainable AI** | SHAP | Surfacing top claim risk factors to human reviewers |
| **Frontend UI** | Streamlit | Queue navigation, details viewer, and analytics dashboards |
| **Deployment** | Docker & Compose | Multi-container environment orchestration |

---

## Getting Started

*(Detailed installation and configuration instructions will be added as implementation progresses.)*

### Prerequisites
- AWS CLI configured with appropriate permissions.
- Docker and Docker Compose installed (optional, for containerized run).
- Python 3.11+ environment.

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone git@github.com:anushacodes/claims-triage-assistant.git
   cd claims-triage-assistant
   ```
2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials and configurations
   ```
3. Prepare the dataset:
   - Place the Kaggle `insurance_claims.csv` dataset in `notebooks/data/insurance_claims.csv`.
