
A small chrome plugin to detect youtube comment sentiments

Project Organization
<div align="center">

# 🎯 YT Comment Sentiment Analysis

### An End-to-End MLOps Pipeline Powering a Chrome Extension for Real-Time YouTube Comment Intelligence

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-API-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![DVC](https://img.shields.io/badge/DVC-Pipeline-945DD6?style=flat-square&logo=dvc&logoColor=white)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20EC2-FF9900?style=flat-square&logo=amazonaws&logoColor=white)](https://aws.amazon.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen?style=flat-square)](./LICENSE)

[Overview](#-overview) • [Architecture](#-system-architecture) • [Features](#-features) • [Tech Stack](#-tech-stack) • [Getting Started](#-getting-started) • [Project Structure](#-project-structure) • [Roadmap](#-roadmap)

</div>

---

## 📌 Overview

**Influencer Insights** is a production-grade, end-to-end machine learning system that classifies the sentiment of YouTube comments in real time and surfaces the results through a lightweight **Chrome extension**. It was built to solve a very real, very expensive problem for high-follower-count creators: **you cannot manually read 50,000 comments — but you can't afford to ignore them either.**

The project goes beyond a Jupyter notebook model — it is engineered as a **complete MLOps pipeline**, covering data versioning, experiment tracking, CI/CD, containerized deployment, and a production Flask inference API consumed by a browser extension.

> Built as part of a hands-on MLOps case study for **Influence Boost Inc.**, an influencer-management platform aiming to attract creators through high-value, low-cost tooling instead of paid marketing.

---

## 🧠 The Problem

| Challenge | Impact |
|---|---|
| High-profile creators receive **thousands of comments per video** | Manual analysis is impossible at scale |
| No time to read through feedback | Missed audience sentiment & content signals |
| Comments are noisy — spam, slang, emojis, sarcasm, multi-language | Naive NLP models fail in production |
| No visibility into sentiment trends over time | Reactive instead of data-driven content strategy |

**The solution:** a Chrome plugin that plugs directly into any YouTube video, pulls the comments via the YouTube Data API, runs them through a trained sentiment classifier, and renders actionable insights — sentiment breakdowns, word clouds, trend lines, and an aggregate engagement score — directly in the browser.

---

## ✨ Features

- 🟢 **Real-Time Sentiment Classification** — Every comment is classified as Positive, Neutral, or Negative on the fly.
- 📊 **Sentiment Distribution Pie Chart** — Instant visual breakdown of audience mood.
- ☁️ **Word Cloud Generation** — Surfaces the most frequently discussed topics and keywords.
- 📈 **Sentiment Trend Graph** — Tracks how sentiment shifts month-over-month across a video's lifetime.
- 🧮 **Average Sentiment Score (0–10)** — A single normalized health metric for quick decision-making.
- 🧹 **Robust Preprocessing** — Handles slang, emojis, informal text, and noisy/spam comments before inference.
- 🔌 **Lightweight Chrome Extension UI** — Insights rendered directly in a popup on the YouTube page, no context-switching required.

---

## 🏗️ System Architecture

The system follows a clean **frontend/backend separation**: the Chrome extension (JavaScript) handles data collection from YouTube and rendering, while a Flask backend owns preprocessing, inference, and chart generation — keeping the extension lightweight and the ML logic centralized and versioned.

```
┌──────────────────────────┐         ┌───────────────────────────────────┐
│      Chrome Extension     │         │            Flask API                │
│  (JavaScript / HTML / CSS)│         │   (Python · scikit-learn/LightGBM)  │
│                            │         │                                     │
│  1. fetchComments(videoId)│  ─────▶ │  /predict                          │
│     via YouTube Data API  │         │  /predict_with_timestamps          │
│                            │ ◀───── │  /generate_chart                   │
│  2. Render popup UI:       │         │  /generate_wordcloud               │
│     • Pie chart            │         │  /generate_trend_graph             │
│     • Word cloud            │         │                                     │
│     • Trend graph           │         │  Preprocess → Vectorize → Predict  │
│     • Avg. sentiment score  │         │  → Aggregate → Plot → Return image │
└──────────────────────────┘         └───────────────────────────────────┘
```

**Request flow for a Sentiment Distribution Pie Chart:**

| Step | Function / Endpoint | Layer | Purpose |
|---|---|---|---|
| 1 | `fetchComments(videoId)` | Frontend | Pull comments via YouTube Data API |
| 2 | `getSentimentPredictions(comments)` → `/predict` | Frontend → Backend | Send comments for inference |
| 3 | `predict()` | Backend | Run the trained model on each comment |
| 4 | Sentiment aggregation (JS) | Frontend | Tally positive / neutral / negative counts |
| 5 | `fetchAndDisplayChart()` → `/generate_chart` | Frontend → Backend | Generate & return chart image |
| 6 | Render in popup | Frontend | Display final visualization |

The same pattern — **fetch → predict → aggregate → visualize → render** — powers the word cloud, trend graph, and average sentiment score features, each with its own dedicated backend endpoint.

---

## 🔬 ML & MLOps Pipeline

This isn't just "train a model and pickle it." The pipeline was designed to be **reproducible, trackable, and production-ready**:

```
Data Collection → Preprocessing → EDA → Model Training & Hyperparameter Tuning
      → DVC Pipeline → Model Registry → Flask API → Chrome Extension
      → CI/CD → Docker → AWS Deployment
```

1. **Data Collection & Preprocessing** — Cleaning noisy, multi-language, slang-heavy, sarcastic, and bot-generated YouTube comments.
2. **EDA** — Understanding class imbalance, comment length distribution, and vocabulary drift.
3. **Model Training** — Classical ML (scikit-learn, LightGBM) with **Optuna** for hyperparameter tuning.
4. **Experiment Tracking** — All runs, metrics, and parameters logged via **MLflow** (tracked through DagsHub).
5. **Data & Pipeline Versioning** — **DVC** manages datasets, pipeline stages (`dvc.yaml`/`dvc.lock`), and artifacts backed by **AWS S3**.
6. **Model Registry** — Promoted models are versioned and staged (staging → production) via the MLflow Model Registry.
7. **Serving** — A **Flask** REST API exposes prediction and visualization endpoints.
8. **CI/CD** — **GitHub Actions** automates testing and deployment on every push.
9. **Containerization & Deployment** — Packaged with **Docker** and deployed to **AWS EC2**, with **CloudWatch** for monitoring.

---

## 🛠️ Tech Stack

<table>
<tr>
<td valign="top" width="33%">

**Machine Learning**
- Python
- scikit-learn
- LightGBM
- NLTK
- Optuna
- Matplotlib · Seaborn · WordCloud

</td>
<td valign="top" width="33%">

**MLOps & Data**
- DVC (Data Version Control)
- MLflow + Model Registry
- DagsHub
- AWS S3

</td>
<td valign="top" width="33%">

**Backend & Deployment**
- Flask (REST API)
- Docker
- GitHub Actions (CI/CD)
- AWS EC2 · CloudWatch

</td>
</tr>
<tr>
<td valign="top" width="33%">

**Frontend (Extension)**
- JavaScript
- HTML / CSS
- Chrome Extension APIs

</td>
<td valign="top" width="33%">

**Testing & Quality**
- Pytest / Unittest
- Pylint

</td>
<td valign="top" width="33%">

**Tooling**
- Git & GitHub
- Postman
- VS Code

</td>
</tr>
</table>

---

## 📂 Project Structure

Built on the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template for a clean, standardized, and reproducible ML project layout:

```
├── .dvc/                    # DVC configuration & remote (S3) settings
├── .github/workflows/       # CI/CD pipelines (GitHub Actions)
├── data/                    # raw → interim → processed → external
├── docs/                    # Sphinx project documentation
├── flask_app/               # Flask REST API — inference & chart generation
├── models/                  # Trained & serialized model artifacts
├── notebooks/               # Exploratory analysis & experimentation
├── references/              # Data dictionaries & explanatory material
├── reports/figures/         # Generated plots & evaluation reports
├── scripts/                 # Utility & automation scripts
├── src/
│   ├── data/                # make_dataset.py — data ingestion
│   ├── features/            # build_features.py — feature engineering
│   ├── models/              # train_model.py, predict_model.py
│   └── visualization/       # visualize.py — plots & charts
├── dockerfile               # Container definition for the Flask API
├── dvc.yaml / dvc.lock       # DVC pipeline stages & reproducibility lock
├── params.yaml               # Centralized model/pipeline hyperparameters
├── requirements.txt
└── setup.py                 # Makes `src` pip-installable
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)
- An AWS account (for S3-backed DVC remote, optional for local runs)

### 1. Clone the repository
```bash
git clone https://github.com/saurav3k2/yt-comment-sentiment-analysis.git
cd yt-comment-sentiment-analysis
```

### 2. Set up the environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Pull versioned data & models (DVC)
```bash
dvc pull
```

### 4. Reproduce the pipeline
```bash
dvc repro
```

### 5. Run the Flask API locally
```bash
python flask_app/app.py
```

### 6. Load the Chrome extension
1. Navigate to `chrome://extensions/`
2. Enable **Developer mode**
3. Click **Load unpacked** and select the extension directory
4. Open any YouTube video and click the extension icon 🎉

### Run with Docker
```bash
docker build -t yt-sentiment-app .
docker run -p 5000:5000 yt-sentiment-app
```

---

## 🧪 Testing

```bash
pytest tests/
```

Unit tests cover preprocessing utilities, model inference, and API endpoints, with linting enforced via **Pylint** and automated on every push via **GitHub Actions**.

---

## 🗺️ Roadmap

- [ ] Multi-language sentiment support
- [ ] Sarcasm-aware model fine-tuning
- [ ] Real-time sentiment drift monitoring / model retraining triggers
- [ ] Export analysis reports as PDF / CSV
- [ ] Auto-scaling deployment via AWS Auto Scaling Groups
- [ ] Prometheus + Grafana dashboards for live monitoring

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](./LICENSE) for more information.

---

<div align="center">

**Built by [Saurav](https://github.com/saurav3k2)** — turning unread comments into actionable content strategy.

⭐ If this project helped you, consider giving it a star!

</div>
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
