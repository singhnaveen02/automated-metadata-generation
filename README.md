# 🔍 Automated Metadata Generator

Upload any PDF, DOCX, TXT, or image file and instantly get:
- Auto-extracted keywords (KeyBERT + TF-IDF)
- Named entity recognition (spaCy)
- AI-generated summary (BART)
- Quality scores and content stats
- JSON export

---

## 📁 Project Structure

```
metadata_app/
├── app.py                  ← Streamlit application
├── requirements.txt        ← Python dependencies
├── packages.txt            ← System-level packages (Tesseract OCR)
├── .streamlit/
│   └── config.toml         ← Upload size & theme settings
└── README.md
```

---

## 🚀 Deploy to GitHub + Streamlit Cloud

### Step 1 — Create a GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Name it `metadata-generator` (or anything you like)
3. Set it to **Public**
4. Click **Create repository**

### Step 2 — Push these files

```bash
# In your local terminal
git init
git add .
git commit -m "Initial commit – metadata generator app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/metadata-generator.git
git push -u origin main
```

### Step 3 — Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **"New app"**
3. Fill in:
   - **Repository:** `YOUR_USERNAME/metadata-generator`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy!"**

Streamlit will automatically install `packages.txt` (Tesseract) and `requirements.txt`.
The first deploy takes ~5–10 minutes due to model downloads.

---

## 💻 Run Locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Launch
streamlit run app.py
```

---

## ⚠️ Notes

- **Tesseract** is required for image OCR. On local Windows, install from
  https://github.com/UB-Mannheim/tesseract/wiki and add to PATH.
- The BART summarization model (~1.6 GB) is downloaded on first run.
  Streamlit Cloud caches it automatically after the first deploy.
- Upload limit is set to 200 MB in `.streamlit/config.toml`.
