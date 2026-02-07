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

A deep learning project to predict the next word in **Nigerian Pidgin** (Naija) text. This system features a dual-model architecture (LSTM + Trigram) served via a FastAPI backend and an interactive Streamlit frontend.

🔗 **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/Jaykay73/nextword-pidgin)  
🔗 **API Docs:** [API Endpoint](https://huggingface.co/spaces/Jaykay73/nextword-pidgin-api/docs)

---

## 📚 Data Collection & Scraping

The foundation of this project is a robust dataset compiled from two primary sources:

### 1. NaijaSenti Dataset
We utilized the **NaijaSenti** dataset (available on Hugging Face), which provides a collection of labeled Nigerian Pidgin tweets and text. This served as our initial baseline corpus.

### 2. Custom BBC Pidgin Scraper
To significantly expand the vocabulary and capture longer-form context, we built a custom web scraper targeting **[BBC News Pidgin](https://www.bbc.com/pidgin)**.

#### How It Works
The scraper (`bbc_pidgin_scraper/scraper.py`) is built using **Python**, **Requests**, and **BeautifulSoup**. The scraping process follows these steps:

1.  **Category Targeting**: We defined key news categories in `config.yml`:
    *   Nigeria (`/topics/c2dwqd1zr92t`)
    *   Africa (`/topics/c404v061z85t`)
    *   Sport (`/topics/cjgn7gv77vrt`)
    *   World (`/topics/c0823e52dd0t`)
    *   Entertainment (`/topics/cqywjyzk2vyt`)

2.  **Pagination Traversal**: The scraper visits each category page and identifies the "Total Pages" span element. It then iterates through all available pages (`/page/1`, `/page/2`, etc.) to maximize coverage.

3.  **URL Extraction**: On each page, it extracts all `<a>` tags. To filter out navigation links and ads, we strictly select URLs that:
    *   Start with `/pidgin/tori`, `/pidgin/world`, or `/pidgin/sport`.
    *   End with a digit (ID), ensuring it's a valid article.

4.  **Content Extraction**: For every valid article URL, the scraper:
    *   Fetches the HTML content.
    *   **Headline**: Extracts text from the `<h1>` tag (or fallback `<strong>` tag).
    *   **Body**: Locates the main story `<div>` using specific CSS classes (e.g., `bbc-19j92fr`). It iterates through all `<p>` tags within this div to compile the full article text.

5.  **Output**: The data is cleaned and saved to `pidgin_corpus.csv` with columns: `headline`, `text`, `category`, `url`.

---

## 🧠 Model Architecture

We implemented two distinct models to provide predictions:

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
*   **Smoothing**: Implements **Laplace Smoothing (Add-1)** to handle unseen N-grams.
*   **Advantages**: Extremely fast and explains common phrases well.

---

## 🛠️ Tech Stack & Deployment

The system uses a decoupled architecture deployed on **Hugging Face Spaces**:

| Component | Technology | Description |
|-----------|------------|-------------|
| **Frontend** | Streamlit | Interactive web UI with real-time inference (using `st_keyup`). |
| **Backend** | FastAPI | REST API serving model predictions. Dockerized for portability. |
| **Environment** | Docker | Ensures consistent runtime with all dependencies (`src` module). |

### Directory Structure
```
.
├── api.py                  # FastAPI application
├── app.py                  # Streamlit frontend
├── bbc_pidgin_scraper/     # Custom web scraper
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
    git clone https://github.com/yourusername/nextword-pidgin.git
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
