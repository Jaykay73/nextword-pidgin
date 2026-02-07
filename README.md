---
title: Nigerian Pidgin Next-Word Predictor
emoji: 🇳🇬
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🇳🇬 Nigerian Pidgin Next-Word Predictor

I built this deep learning project to predict the next word in **Nigerian Pidgin** (Naija) text. The system features a dual-model architecture (LSTM + Trigram) served via a FastAPI backend and an interactive Streamlit frontend.

🔗 **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/Jaykay73/nextword-pidgin)  
🔗 **API Docs:** [API Endpoint](https://huggingface.co/spaces/Jaykay73/nextword-pidgin-api/docs)

---

## 📚 Data Collection & Scraping

The foundation of my project is a robust dataset compiled from two primary sources:

### 1. NaijaSenti Dataset
I utilized the **NaijaSenti** dataset (available on Hugging Face), which provides a collection of labeled Nigerian Pidgin tweets and text. This served as my initial baseline corpus.

### 2. BBC Pidgin Scraper
To significantly expand the vocabulary and capture longer-form context, I utilized an existing scraper tool by **[keleog](https://github.com/keleog/bbc_pidgin_scraper)**.

#### How It Works
I cloned the `bbc_pidgin_scraper` repository to fetch articles from **[BBC News Pidgin](https://www.bbc.com/pidgin)**. The scraper (built with Python, Requests, and BeautifulSoup) allowed me to target specific categories:

1.  **Category Targeting**: It targeted key news sections:
    *   Nigeria (`/topics/c2dwqd1zr92t`)
    *   Africa (`/topics/c404v061z85t`)
    *   Sport (`/topics/cjgn7gv77vrt`)
    *   World (`/topics/c0823e52dd0t`)
    *   Entertainment (`/topics/cqywjyzk2vyt`)

2.  **Data Extraction**: The tool traversed pagination to collect article URLs and then extracted the **headlines** and **body text** from each page, filtering out ads and navigation links.

3.  **Result**: I compiled this data into `pidgin_corpus.csv`, providing a rich source of authentic Pidgin text for training my models.

---

## 🧠 Model Architecture

I implemented two distinct models to provide predictions:

### 1. LSTM (Long Short-Term Memory)
A neural network designed to capture long-range dependencies in text.
*   **Embedding Layer**: 256 dimensions.
*   **Hidden Layers**: 2 stacked LSTM layers with 512 units each.
*   **Dropout**: 0.3 for regularization.
*   **Context Window**: 15 words.
*   **Framework**: PyTorch.

### 2. Trigram Model (Statistical)
A baseline N-gram model using `N-1` Markov assumption.
*   **Logic**: Calculates `P(w3 | w1, w2)`.
*   **Smoothing**: I implemented **Laplace Smoothing (Add-1)** to handle unseen N-grams.
*   **Advantages**: Extremely fast and explains common phrases well.

---

## 🛠️ Tech Stack & Deployment

My system uses a decoupled architecture deployed on **Hugging Face Spaces**:

| Component | Technology | Description |
|-----------|------------|-------------|
| **Frontend** | Streamlit | Interactive web UI with real-time inference (using `st_keyup`). |
| **Backend** | FastAPI | REST API serving model predictions. Dockerized for portability. |
| **Environment** | Docker | Ensures consistent runtime for the API. |

### Directory Structure
```
.
├── api.py                  # FastAPI application
├── app.py                  # Streamlit frontend
├── bbc_pidgin_scraper/     # Scraper module (adapted from keleog)
├── docker/                 # Deployment configurations
├── notebooks/              # Training notebooks (LSTM)
├── src/                    # Shared source code
│   ├── data_loader.py      # Data ingestion logic
│   ├── preprocessing.py    # Text cleaning & tokenization
│   └── trigram_model.py    # Trigram logic
└── requirements.txt        # Project dependencies
```

---

## 🚀 Running Locally

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Jaykay73/nextword-pidgin.git
    cd nextword-pidgin
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the API**:
    ```bash
    uvicorn api:app --reload
    ```
    API will run at `http://localhost:8000`.

4.  **Run the Frontend**:
    ```bash
    streamlit run app.py
    ```
    App will open at `http://localhost:8501`.

---

## 📝 Usage

1.  Open the web app.
2.  Type a phrase in Nigerian Pidgin (e.g., *"How far"*).
3.  Click on a suggested word to append it to your text.
4.  Toggle between **LSTM** and **Trigram** models to see different prediction styles.
