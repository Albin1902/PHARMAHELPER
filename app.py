import os
import re
from typing import Dict, List

import pandas as pd
import streamlit as st


# -----------------------------
# Config
# -----------------------------
DEFAULT_DATA_PATHS = [
    # root (YOUR CURRENT REPO STRUCTURE)
    "medications_enriched.csv",
    "medications_enriched.xlsx",
    "medications.xlsx",

    # optional /data folder
    "data/medications_enriched.csv",
    "data/medications_enriched.xlsx",
    "data/medications.xlsx",

    # legacy names
    "data/sdm_medications_enriched_offline.xlsx",
    "data/sdm_medications_enriched_offline.csv",
]

SEARCH_COLUMNS = [
    "brand_name",
    "DIN",
    "generic_name",
    "category",
    "form",
    "synonyms",
]


# -----------------------------
# Query Expansion
# -----------------------------
PHRASE_MAP: Dict[str, List[str]] = {
    "blue inhaler": ["salbutamol", "ventolin", "rescue", "puffer", "hfa", "inhaler"],
    "rescue inhaler": ["salbutamol", "ventolin", "puffer", "hfa"],
    "water pill": ["diuretic", "furosemide", "hydrochlorothiazide", "spironolactone"],
    "fluid pill": ["diuretic", "furosemide"],
    "insulin pen": ["pen", "injection", "ozempic", "wegovy"],
    "nasal spray": ["spray", "nasal"],
}

TOKEN_MAP: Dict[str, List[str]] = {
    "bp": ["blood pressure", "hypertension"],
    "uti": ["urinary", "infection"],
    "cholesterol": ["statin", "lipid"],
    "blood": ["anticoagulant", "clot", "stroke"],
    "allergy": ["hay fever", "antihistamine"],
}


# -----------------------------
# Helpers
# -----------------------------
def find_data_path() -> str:
    for p in DEFAULT_DATA_PATHS:
        if os.path.exists(p):
            return p

    env_path = os.environ.get("SDM_DATA_PATH", "").strip()
    if env_path and os.path.exists(env_path):
        return env_path

    return ""


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not path:
        raise FileNotFoundError(
            "No data file found.\n\n"
            "Expected one of:\n" + "\n".join(DEFAULT_DATA_PATHS)
        )

    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, dtype={"DIN": str})
    else:
        try:
            df = pd.read_csv(path, dtype={"DIN": str}, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype={"DIN": str}, encoding="cp1252")

    for col in SEARCH_COLUMNS:
        if col not in df.columns:
            df[col] = ""

        df[col] = (
            df[col]
            .astype(str)
            .replace({"nan": "", "None": ""})
            .fillna("")
        )

    df["DIN"] = df["DIN"].apply(lambda x: re.sub(r"\D", "", str(x)))

    return df


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def expand_query(q: str) -> str:
    qn = normalize(q)
    extra = []

    for phrase, words in PHRASE_MAP.items():
        if phrase in qn:
            extra.extend(words)

    for token in re.split(r"[,\s/]+", qn):
        if token in TOKEN_MAP:
            extra.extend(TOKEN_MAP[token])

    return " ".join([qn] + extra)


def row_blob(row: pd.Series) -> str:
    return normalize(" | ".join(str(row[c]) for c in SEARCH_COLUMNS))


def matches(blob: str, query: str) -> bool:
    if not query:
        return True
    return all(w in blob for w in query.split())


def missing_info(row: pd.Series) -> bool:
    return any(not row[c].strip() for c in ["generic_name", "category", "form"])


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="SDM Medication Navigator", layout="wide")

st.title("💊 SDM Medication Navigator")
st.caption("Offline lookup • Search by brand / generic / category / form / synonyms")

data_path = find_data_path()

with st.sidebar:
    st.header("Filters")

    st.write("**Data file detected**")
    st.code(data_path if data_path else "NOT FOUND")

    override = st.text_input("Override path (optional)")
    if override.strip():
        data_path = override.strip()

    if not data_path:
        st.error("No data file found. Add CSV/XLSX to repo root or /data.")
        st.stop()

    df = load_data(data_path)

    categories = ["All"] + sorted(c for c in df["category"].unique() if c.strip())
    forms = ["All"] + sorted(f for f in df["form"].unique() if f.strip())

    cat = st.selectbox("Category", categories)
    form = st.selectbox("Form", forms)

    only_missing = st.checkbox("Show only rows missing info")


search = st.text_input(
    "Search (brand / generic / category / form / synonyms)",
    placeholder="blue inhaler, water pill, apixaban, cholesterol",
)

expanded = expand_query(search)

filtered = df.copy()

if cat != "All":
    filtered = filtered[filtered["category"] == cat]

if form != "All":
    filtered = filtered[filtered["form"] == form]

if only_missing:
    filtered = filtered[filtered.apply(missing_info, axis=1)]

if expanded:
    blobs = filtered.apply(row_blob, axis=1)
    filtered = filtered[blobs.apply(lambda b: matches(b, expanded))]

st.subheader(f"Results ({len(filtered)})")

display_cols = [
    "brand_name",
    "generic_name",
    "DIN",
    "category",
    "form",
    "synonyms",
]

st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

st.download_button(
    "Download current results as CSV",
    filtered[display_cols].to_csv(index=False).encode("utf-8-sig"),
    "sdm_medications_filtered.csv",
    "text/csv",
)
