import streamlit as st
import json
import os
import re
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Automated Metadata Generator",
    page_icon="🔍",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# Lazy-load heavy libraries so the app starts quickly
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NLP models…")
def load_models():
    import spacy
    import nltk
    from keybert import KeyBERT
    from transformers import pipeline

    nltk.download("punkt",       quiet=True)
    nltk.download("punkt_tab",   quiet=True)
    nltk.download("stopwords",   quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download as spacy_download
        spacy_download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    kw_model = KeyBERT()

    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception:
        summarizer = None

    return nlp, kw_model, summarizer


# ─────────────────────────────────────────────────────────────
# Text extraction helpers
# ─────────────────────────────────────────────────────────────
def extract_text(file_path: str, ext: str):
    """Return (raw_text, extraction_metadata_dict)."""
    import pandas as pd

    ext = ext.lower()

    if ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext == ".docx":
        return _extract_docx(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        meta = {"word_count": len(text.split()), "character_count": len(text)}
        return text, meta
    elif ext in (".png", ".jpg", ".jpeg"):
        return _extract_ocr(file_path)
    return "", {}


def _extract_pdf(path):
    import fitz, pdfplumber

    # PyMuPDF
    pymeta, py_text = {}, ""
    try:
        doc = fitz.open(path)
        pymeta = {
            "title":  doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "page_count": doc.page_count,
        }
        py_text = "\n".join(
            doc.load_page(i).get_text()
            for i in range(doc.page_count)
        )
        doc.close()
    except Exception:
        pass

    # pdfplumber fallback
    pb_text = ""
    try:
        with pdfplumber.open(path) as pdf:
            pb_text = "\n".join(
                p.extract_text() or "" for p in pdf.pages
            )
    except Exception:
        pass

    text = py_text if len(py_text) >= len(pb_text) else pb_text
    return text, pymeta


def _extract_docx(path):
    import docx2txt
    text = docx2txt.process(path) or ""
    meta = {"word_count": len(text.split()), "character_count": len(text)}
    return text, meta


def _extract_ocr(path):
    import cv2, pytesseract
    from PIL import Image

    img = cv2.imread(path)
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    processed = 255 - opened

    text = pytesseract.image_to_string(processed, config="--psm 6", lang="eng")
    with Image.open(path) as im:
        meta = {"image_size": im.size, "image_mode": im.mode}
    return text, meta


# ─────────────────────────────────────────────────────────────
# Core processing
# ─────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)]", " ", text)
    text = re.sub(r"\bPage\s+\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"http\S+", "", text)
    return text.strip()


def extract_keywords(text: str, kw_model, n=12):
    if not text or len(text) < 50:
        return []
    try:
        kws = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_k=n,
            use_mmr=True,
            diversity=0.5,
        )
        return [{"keyword": k, "confidence": round(s, 3)} for k, s in kws]
    except Exception:
        return []


def extract_entities(text: str, nlp):
    if not text:
        return {}
    doc = nlp(text)
    buckets: dict = {}
    for ent in doc.ents:
        if len(ent.text.strip()) < 2:
            continue
        buckets.setdefault(ent.label_, {})
        buckets[ent.label_][ent.text.strip()] = \
            buckets[ent.label_].get(ent.text.strip(), 0) + 1

    mapping = {
        "PERSON": "people",
        "ORG": "organizations",
        "GPE": "locations",
        "LOC": "locations",
        "DATE": "dates",
        "TIME": "dates",
    }
    result: dict = {
        "people": [], "organizations": [],
        "locations": [], "dates": [], "other": {}
    }
    for label, counts in buckets.items():
        top = sorted(counts, key=counts.get, reverse=True)[:8]
        cat = mapping.get(label)
        if cat:
            result[cat].extend(top)
        else:
            result["other"][label] = top
    return result


def summarize(text: str, summarizer) -> str:
    if not text or summarizer is None:
        return ""
    from nltk.tokenize import sent_tokenize

    chunks, buf, buf_len = [], [], 0
    for s in sent_tokenize(text):
        if buf_len + len(s) > 1024 and buf:
            chunks.append(" ".join(buf))
            buf, buf_len = [s], len(s)
        else:
            buf.append(s)
            buf_len += len(s)
    if buf:
        chunks.append(" ".join(buf))

    summaries = []
    for chunk in chunks[:3]:
        try:
            out = summarizer(chunk, max_length=150, min_length=40,
                             do_sample=False, num_beams=4, early_stopping=True)
            summaries.append(out[0]["summary_text"])
        except Exception:
            pass
    return " ".join(summaries)


def build_metadata(text: str, raw_meta: dict, filename: str,
                   nlp, kw_model, summarizer) -> dict:
    import pandas as pd
    from nltk.tokenize import sent_tokenize, word_tokenize

    words     = text.split()
    sentences = sent_tokenize(text) if text else []
    unique    = set(w.lower() for w in words if w.isalpha())

    keywords = extract_keywords(text, kw_model)
    entities = extract_entities(text, nlp)
    summary  = summarize(text, summarizer)

    avg_wps = len(words) / len(sentences) if sentences else 0
    complexity = "high" if avg_wps > 20 else ("medium" if avg_wps > 15 else "low")

    # Quality scores
    ext_conf  = min(0.4 * (len(text) > 1000) + 0.3 * (len(words) > 0) + 0.3 * bool(raw_meta.get("title")), 1.0)
    txt_qual  = min(0.4 * (len(text) > 500) + 0.3 * (len(words) > 100) + 0.3, 1.0)
    comp_n    = sum([bool(keywords), bool(any(entities.values())), len(text) > 200, len(words) > 50])
    comp_sc   = comp_n / 4

    return {
        "document_info": {
            "title":           raw_meta.get("title") or Path(filename).stem.replace("_", " ").title(),
            "filename":        filename,
            "file_type":       Path(filename).suffix,
            "author":          raw_meta.get("author", ""),
            "page_count":      raw_meta.get("page_count", ""),
            "processing_date": pd.Timestamp.now().isoformat(),
        },
        "content_analysis": {
            "language":        "en",
            "word_count":      len(words),
            "character_count": len(text),
            "sentence_count":  len(sentences),
            "unique_words":    len(unique),
            "vocabulary_richness": round(len(unique) / len(words), 3) if words else 0,
            "avg_words_per_sentence": round(avg_wps, 2),
            "complexity_level": complexity,
        },
        "semantic_metadata": {
            "keywords":      [k["keyword"] for k in keywords],
            "keyword_details": keywords,
            "summary":       summary,
        },
        "entities":        entities,
        "quality_metrics": {
            "extraction_confidence": round(ext_conf, 2),
            "text_quality_score":    round(txt_qual, 2),
            "completeness_score":    round(comp_sc, 2),
        },
    }


# ─────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────
st.title("🔍 Automated Metadata Generator")
st.markdown(
    "Upload a **PDF, DOCX, TXT, JPG or PNG** document. "
    "The app extracts keywords, named entities, a summary, "
    "quality scores, and exports everything as JSON."
)

# Load models once
nlp, kw_model, summarizer = load_models()

# ── Upload ──────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📁 Upload your document",
    type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
)

if uploaded:
    st.success(f"✅ File ready: **{uploaded.name}**  ({uploaded.size / 1024:.1f} KB)")

    if st.button("⚡ Process Document", type="primary"):
        # Save to temp file
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Extracting text…"):
                raw_text, raw_meta = extract_text(tmp_path, suffix)

            if not raw_text.strip():
                st.error("❌ Could not extract any text from this file.")
                st.stop()

            with st.spinner("Preprocessing…"):
                clean_text = preprocess(raw_text)

            with st.spinner("Analysing (keywords · entities · summary)…"):
                metadata = build_metadata(
                    clean_text, raw_meta, uploaded.name,
                    nlp, kw_model, summarizer
                )

        finally:
            os.remove(tmp_path)

        # ── Results ─────────────────────────────────────────
        st.success("🎉 Processing complete!")

        col1, col2, col3 = st.columns(3)
        q = metadata["quality_metrics"]
        col1.metric("Extraction Confidence", f"{q['extraction_confidence']:.0%}")
        col2.metric("Text Quality",           f"{q['text_quality_score']:.0%}")
        col3.metric("Completeness",           f"{q['completeness_score']:.0%}")

        # Document info
        st.subheader("📄 Document Info")
        di = metadata["document_info"]
        info_cols = st.columns(3)
        info_cols[0].markdown(f"**Title:** {di['title']}")
        info_cols[1].markdown(f"**Type:** {di['file_type']}")
        info_cols[2].markdown(f"**Author:** {di['author'] or '—'}")

        # Content stats
        st.subheader("📊 Content Analysis")
        ca = metadata["content_analysis"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Words",         f"{ca['word_count']:,}")
        c2.metric("Sentences",     f"{ca['sentence_count']:,}")
        c3.metric("Unique Words",  f"{ca['unique_words']:,}")
        c4.metric("Complexity",    ca["complexity_level"].title())

        # Summary
        if metadata["semantic_metadata"]["summary"]:
            st.subheader("📝 Auto-Generated Summary")
            st.info(metadata["semantic_metadata"]["summary"])

        # Keywords
        st.subheader("🏷️ Top Keywords")
        kw_details = metadata["semantic_metadata"]["keyword_details"]
        if kw_details:
            import pandas as pd
            kw_df = pd.DataFrame(kw_details)
            st.dataframe(
                kw_df.rename(columns={"keyword": "Keyword", "confidence": "Confidence"}),
                use_container_width=True,
                hide_index=True,
            )

        # Entities
        st.subheader("👥 Named Entities")
        ents = metadata["entities"]
        ec1, ec2, ec3, ec4 = st.columns(4)
        for col, label, key in [
            (ec1, "👤 People",        "people"),
            (ec2, "🏢 Organizations", "organizations"),
            (ec3, "📍 Locations",     "locations"),
            (ec4, "📅 Dates",         "dates"),
        ]:
            col.markdown(f"**{label}**")
            items = ents.get(key, [])
            col.write(", ".join(items[:6]) if items else "—")

        # Full JSON preview
        with st.expander("🔎 Full Metadata JSON"):
            st.json(metadata)

        # Download button
        json_str = json.dumps(metadata, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            label="⬇️ Download Metadata (JSON)",
            data=json_str,
            file_name=f"metadata_{Path(uploaded.name).stem}.json",
            mime="application/json",
        )

else:
    st.info("👆 Upload a file above to get started.")
