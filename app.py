import streamlit as st
import json
import os
import re
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Automated Metadata Generator",
    page_icon="🔍",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# Model loading — lazy, cached, with clear error messages
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NLP models… (first run takes ~60 s)")
def load_models():
    import nltk
    import spacy
    from keybert import KeyBERT

    for pkg in ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger"):
        nltk.download(pkg, quiet=True)

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download as spacy_dl
        spacy_dl("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    kw_model = KeyBERT()

    # ── Use sumy for summarisation (lightweight, no torch needed) ──
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lsa import LsaSummarizer
        summarizer_type = "sumy"
    except ImportError:
        summarizer_type = "extractive"   # pure-Python fallback

    return nlp, kw_model, summarizer_type


# ─────────────────────────────────────────────────────────────
# Text extraction
# ─────────────────────────────────────────────────────────────
def extract_text(file_path: str, ext: str):
    ext = ext.lower()
    if ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext == ".docx":
        return _extract_docx(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return text, {"word_count": len(text.split()), "character_count": len(text)}
    elif ext in (".png", ".jpg", ".jpeg"):
        return _extract_ocr(file_path)
    return "", {}


def _extract_pdf(path):
    import fitz, pdfplumber
    py_text, meta = "", {}
    try:
        doc = fitz.open(path)
        meta = {
            "title":      doc.metadata.get("title", ""),
            "author":     doc.metadata.get("author", ""),
            "page_count": doc.page_count,
        }
        py_text = "\n".join(doc.load_page(i).get_text() for i in range(doc.page_count))
        doc.close()
    except Exception:
        pass
    pb_text = ""
    try:
        with pdfplumber.open(path) as pdf:
            pb_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:
        pass
    best = py_text if len(py_text) >= len(pb_text) else pb_text
    return best, meta


def _extract_docx(path):
    import docx2txt
    text = docx2txt.process(path) or ""
    return text, {"word_count": len(text.split()), "character_count": len(text)}


def _extract_ocr(path):
    import cv2, pytesseract
    from PIL import Image
    img    = cv2.imread(path)
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    text = pytesseract.image_to_string(255 - opened, config="--psm 6", lang="eng")
    with Image.open(path) as im:
        meta = {"image_size": str(im.size), "image_mode": im.mode}
    return text, meta


# ─────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)]", " ", text)
    text = re.sub(r"\bPage\s+\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"http\S+", "", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────
# Keyword extraction
# ─────────────────────────────────────────────────────────────
def extract_keywords(text: str, kw_model, n: int = 12):
    if not text or len(text) < 50:
        return []
    try:
        kws = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=n,
            use_mmr=True,
            diversity=0.5,
        )
        return [{"keyword": k, "confidence": round(s, 3)} for k, s in kws]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────
# Named entity recognition
# ─────────────────────────────────────────────────────────────
def extract_entities(text: str, nlp):
    if not text or nlp is None:
        return {}
    mapping = {
        "PERSON": "people", "ORG": "organizations",
        "GPE": "locations", "LOC": "locations",
        "DATE": "dates",    "TIME": "dates",
    }
    buckets: dict = {}
    for ent in nlp(text).ents:
        t = ent.text.strip()
        if len(t) < 2:
            continue
        buckets.setdefault(ent.label_, {})
        buckets[ent.label_][t] = buckets[ent.label_].get(t, 0) + 1

    result: dict = {"people": [], "organizations": [], "locations": [], "dates": [], "other": {}}
    for label, counts in buckets.items():
        top = sorted(counts, key=counts.get, reverse=True)[:8]
        cat = mapping.get(label)
        if cat:
            result[cat].extend(top)
        else:
            result["other"][label] = top
    return result


# ─────────────────────────────────────────────────────────────
# Summarisation  (sumy LSA — no torch, no GPU needed)
# ─────────────────────────────────────────────────────────────
def summarize(text: str, summarizer_type: str, n_sentences: int = 5) -> str:
    if not text or len(text.split()) < 30:
        return ""
    if summarizer_type == "sumy":
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.lsa import LsaSummarizer
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LsaSummarizer()
            sentences = summarizer(parser.document, n_sentences)
            return " ".join(str(s) for s in sentences)
        except Exception:
            pass
    # Pure-Python extractive fallback
    from nltk.tokenize import sent_tokenize
    sents = sent_tokenize(text)
    if not sents:
        return ""
    pick = sents[:3] + (sents[-2:] if len(sents) > 5 else [])
    return " ".join(pick[:n_sentences])


# ─────────────────────────────────────────────────────────────
# Metadata assembly
# ─────────────────────────────────────────────────────────────
def build_metadata(text: str, raw_meta: dict, filename: str,
                   nlp, kw_model, summarizer_type: str) -> dict:
    import pandas as pd
    from nltk.tokenize import sent_tokenize

    words     = text.split()
    sentences = sent_tokenize(text) if text else []
    unique    = set(w.lower() for w in words if w.isalpha())
    avg_wps   = len(words) / len(sentences) if sentences else 0
    complexity = "high" if avg_wps > 20 else ("medium" if avg_wps > 15 else "low")

    keywords = extract_keywords(text, kw_model)
    entities = extract_entities(text, nlp)
    summary  = summarize(text, summarizer_type)

    n_kw   = len(keywords)
    n_ents = sum(len(v) for v in entities.values() if isinstance(v, list))
    ext_c  = min(0.4*(len(text)>1000) + 0.3*(len(words)>0) + 0.3*bool(raw_meta.get("title")), 1.0)
    tq     = min(0.4*(len(text)>500)  + 0.3*(len(words)>100) + 0.3, 1.0)
    comp_s = sum([n_kw>0, n_ents>0, len(text)>200, len(words)>50]) / 4

    return {
        "document_info": {
            "title":           raw_meta.get("title") or Path(filename).stem.replace("_"," ").title(),
            "filename":        filename,
            "file_type":       Path(filename).suffix,
            "author":          raw_meta.get("author", ""),
            "page_count":      raw_meta.get("page_count", ""),
            "processing_date": pd.Timestamp.now().isoformat(),
        },
        "content_analysis": {
            "language":              "en",
            "word_count":            len(words),
            "character_count":       len(text),
            "sentence_count":        len(sentences),
            "unique_words":          len(unique),
            "vocabulary_richness":   round(len(unique)/len(words), 3) if words else 0,
            "avg_words_per_sentence":round(avg_wps, 2),
            "complexity_level":      complexity,
        },
        "semantic_metadata": {
            "keywords":        [k["keyword"] for k in keywords],
            "keyword_details": keywords,
            "summary":         summary,
        },
        "entities":        entities,
        "quality_metrics": {
            "extraction_confidence": round(ext_c,  2),
            "text_quality_score":    round(tq,     2),
            "completeness_score":    round(comp_s, 2),
        },
    }


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.title("🔍 Automated Metadata Generator")
st.markdown(
    "Upload a **PDF, DOCX, TXT, JPG or PNG** — get keywords, entities, "
    "summary, quality scores, and a JSON export."
)

# Load models once
try:
    nlp, kw_model, summarizer_type = load_models()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

uploaded = st.file_uploader(
    "📁 Upload your document",
    type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
)

if uploaded:
    st.success(f"✅ **{uploaded.name}**  ({uploaded.size/1024:.1f} KB)")

    if st.button("⚡ Process Document", type="primary"):
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Extracting text…"):
                raw_text, raw_meta = extract_text(tmp_path, suffix)

            if not raw_text.strip():
                st.error("❌ No text could be extracted.")
                st.stop()

            with st.spinner("Preprocessing…"):
                clean = preprocess(raw_text)

            with st.spinner("Analysing keywords, entities & summary…"):
                metadata = build_metadata(clean, raw_meta, uploaded.name,
                                          nlp, kw_model, summarizer_type)
        finally:
            os.remove(tmp_path)

        st.success("🎉 Done!")

        # ── Quality metrics row ──────────────────────────────
        q = metadata["quality_metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Extraction Confidence", f"{q['extraction_confidence']:.0%}")
        c2.metric("Text Quality",           f"{q['text_quality_score']:.0%}")
        c3.metric("Completeness",           f"{q['completeness_score']:.0%}")

        # ── Document info ────────────────────────────────────
        st.subheader("📄 Document Info")
        di = metadata["document_info"]
        r1, r2, r3 = st.columns(3)
        r1.markdown(f"**Title:** {di['title']}")
        r2.markdown(f"**Type:** {di['file_type']}  |  **Pages:** {di.get('page_count','—')}")
        r3.markdown(f"**Author:** {di['author'] or '—'}")

        # ── Content stats ────────────────────────────────────
        st.subheader("📊 Content Analysis")
        ca = metadata["content_analysis"]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Words",        f"{ca['word_count']:,}")
        s2.metric("Sentences",    f"{ca['sentence_count']:,}")
        s3.metric("Unique Words", f"{ca['unique_words']:,}")
        s4.metric("Complexity",   ca["complexity_level"].title())

        # ── Summary ──────────────────────────────────────────
        sm = metadata["semantic_metadata"]
        if sm["summary"]:
            st.subheader("📝 Summary")
            st.info(sm["summary"])

        # ── Keywords ─────────────────────────────────────────
        st.subheader("🏷️ Top Keywords")
        if sm["keyword_details"]:
            import pandas as pd
            st.dataframe(
                pd.DataFrame(sm["keyword_details"])
                  .rename(columns={"keyword":"Keyword","confidence":"Confidence"}),
                use_container_width=True,
                hide_index=True,
            )

        # ── Entities ─────────────────────────────────────────
        st.subheader("👥 Named Entities")
        ents = metadata["entities"]
        e1, e2, e3, e4 = st.columns(4)
        for col, label, key in [
            (e1, "👤 People",        "people"),
            (e2, "🏢 Organizations", "organizations"),
            (e3, "📍 Locations",     "locations"),
            (e4, "📅 Dates",         "dates"),
        ]:
            col.markdown(f"**{label}**")
            items = ents.get(key, [])
            col.write(", ".join(items[:6]) if items else "—")

        # ── Full JSON ────────────────────────────────────────
        with st.expander("🔎 Full Metadata JSON"):
            st.json(metadata)

        # ── Download ─────────────────────────────────────────
        st.download_button(
            label="⬇️ Download Metadata JSON",
            data=json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
            file_name=f"metadata_{Path(uploaded.name).stem}.json",
            mime="application/json",
        )
else:
    st.info("👆 Upload a file above to get started.")
