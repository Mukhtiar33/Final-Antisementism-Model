"""
preprocess.py — Production-ready YouTube NLP Preprocessing Pipeline
Handles messy scraped Excel files with metadata rows before actual data.
"""

import os
import re
import logging
import warnings
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Try importing VADER ──────────────────────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    log.warning("vaderSentiment not installed. Run: pip install vaderSentiment")
    VADER_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────────────
GARBAGE_KEYWORDS = {
    "video id", "video title", "video url", "channel name", "channel url",
    "upload date", "video duration", "view count", "like count", "description",
    "comment_id", "url sequence",
}
NUMERIC_SEQ_PATTERN = re.compile(r"^(\d+\s+){5,}")  # "1 2 3 4 5 ..." pattern
CLEANING_PATTERN = re.compile(
    r"http\S+|www\.\S+|"       # URLs
    r"@\w+|"                    # Mentions
    r"#\w+|"                    # Hashtags
    r"[^\w\s]|"                # Special chars
    r"\s+"                      # Extra whitespace
)

# ── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"#\w+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_garbage_row(text: str) -> bool:
    """Return True if the row looks like metadata or a scraping artifact."""
    if not isinstance(text, str):
        return True
    t = text.lower().strip()
    if not t:
        return True
    if any(kw in t for kw in GARBAGE_KEYWORDS):
        return True
    if NUMERIC_SEQ_PATTERN.match(t):
        return True
    # >60% tokens are numeric
    tokens = t.split()
    if len(tokens) == 0:
        return True
    numeric_ratio = sum(1 for tok in tokens if tok.isdigit()) / len(tokens)
    if numeric_ratio > 0.6:
        return True
    # Too short
    if len(tokens) < 2:
        return True
    return False

# ── Header Detection ─────────────────────────────────────────────────────────

COMMENT_HEADER_HINTS = {"comment", "comment_text", "text", "content", "body"}
TRANSCRIPT_HEADER_HINTS = {"transcript", "transcript_text", "caption", "text", "content"}


def detect_header_row(filepath: str, hints: set, max_scan: int = 25) -> int:
    """
    Scan up to max_scan rows to find the real header row.
    Returns the 0-based row index to pass as `header=` to pd.read_excel/csv.
    Falls back to 0 if nothing found.
    """
    try:
        ext = Path(filepath).suffix.lower()
        raw = (
            pd.read_excel(filepath, header=None, nrows=max_scan)
            if ext in (".xlsx", ".xls")
            else pd.read_csv(filepath, header=None, nrows=max_scan)
        )
    except Exception as e:
        log.warning(f"Cannot scan {filepath}: {e}")
        return 0

    for i, row in raw.iterrows():
        row_vals = {str(v).lower().strip() for v in row if pd.notna(v)}
        if row_vals & hints:
            log.info(f"  Header detected at row {i} in {Path(filepath).name}")
            return i
    log.warning(f"  No header found in {Path(filepath).name}, defaulting to row 0")
    return 0

# ── File Loading ─────────────────────────────────────────────────────────────

def load_file(filepath: str, header_hints: set) -> pd.DataFrame | None:
    ext = Path(filepath).suffix.lower()
    header_row = detect_header_row(filepath, header_hints)
    try:
        df = (
            pd.read_excel(filepath, header=header_row)
            if ext in (".xlsx", ".xls")
            else pd.read_csv(filepath, header=header_row)
        )
        # Normalize column names
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        return df
    except Exception as e:
        log.warning(f"  Failed to load {filepath}: {e}")
        return None

# ── Column Resolution ────────────────────────────────────────────────────────

def resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ── Sentiment ────────────────────────────────────────────────────────────────

def get_sentiment(text: str) -> tuple[float, int]:
    if not VADER_AVAILABLE or not isinstance(text, str) or not text.strip():
        return 0.0, 0
    score = analyzer.polarity_scores(text)["compound"]
    label = 1 if score >= 0.05 else (-1 if score <= -0.05 else 0)
    return round(score, 4), label

# ── Transcript Processing ────────────────────────────────────────────────────

def process_transcript(filepath: str, video_id: str) -> str:
    df = load_file(filepath, TRANSCRIPT_HEADER_HINTS)
    if df is None:
        return ""

    text_col = resolve_column(df, ["transcript_text", "text", "caption", "content", "transcript"])
    if text_col is None:
        log.warning(f"  No text column found in transcript for {video_id}")
        return ""

    chunks = []
    for val in df[text_col].dropna():
        t = str(val).strip()
        if not t or NUMERIC_SEQ_PATTERN.match(t.lower()):
            continue
        if any(kw in t.lower() for kw in GARBAGE_KEYWORDS):
            continue
        chunks.append(t)

    full = " ".join(chunks)
    # Remove leftover numeric sequences mid-text
    full = re.sub(r"\b\d+\b(\s+\d+\b){4,}", " ", full)
    full = re.sub(r"\s+", " ", full).strip()
    return full[:3000]  # cap at 3000 chars

# ── Comment Processing ───────────────────────────────────────────────────────

def process_comments(filepath: str, video_id: str) -> pd.DataFrame:
    df = load_file(filepath, COMMENT_HEADER_HINTS)
    if df is None:
        return pd.DataFrame()

    text_col = resolve_column(df, ["comment_text", "text", "content", "body", "comment"])
    if text_col is None:
        log.warning(f"  No text column found in comments for {video_id}")
        return pd.DataFrame()

    rows = []
    for val in df[text_col].dropna():
        raw = str(val).strip()
        if is_garbage_row(raw):
            continue
        cleaned = clean_text(raw)
        if not cleaned or len(cleaned.split()) < 2:
            continue
        score, label = get_sentiment(cleaned)
        rows.append({
            "video_id": video_id,
            "comment_text": raw,
            "clean_text": cleaned,
            "sentiment_score": score,
            "sentiment_label": label,
            "comment_length": len(raw),
            "word_count": len(cleaned.split()),
        })

    return pd.DataFrame(rows)

# ── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(base_dir: str):
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    all_comments: list[pd.DataFrame] = []
    transcript_dict: dict[str, str] = {}
    stats = {"videos": 0, "comments_kept": 0, "comments_removed": 0, "videos_no_transcript": 0}

    video_dirs = [d for d in base.iterdir() if d.is_dir()]
    if not video_dirs:
        log.error("No video subdirectories found.")
        return

    for video_dir in sorted(video_dirs):
        video_id = video_dir.name
        log.info(f"Processing: {video_id}")
        stats["videos"] += 1

        # ── Find files ───────────────────────────────────────────────────────
        comment_file = transcript_file = None
        for f in video_dir.iterdir():
            name = f.name.lower()
            if "comment" in name and f.suffix.lower() in (".xlsx", ".xls", ".csv"):
                comment_file = str(f)
            elif "transcript" in name and f.suffix.lower() in (".xlsx", ".xls", ".csv"):
                transcript_file = str(f)

        # ── Transcript ───────────────────────────────────────────────────────
        if transcript_file:
            transcript_dict[video_id] = process_transcript(transcript_file, video_id)
        else:
            log.warning(f"  No transcript file found for {video_id}")
            transcript_dict[video_id] = ""
            stats["videos_no_transcript"] += 1

        # ── Comments ─────────────────────────────────────────────────────────
        if comment_file:
            before = sum(1 for _ in open(comment_file, errors="ignore"))
            df = process_comments(comment_file, video_id)
            stats["comments_kept"] += len(df)
            stats["comments_removed"] += max(0, before - len(df))
            if not df.empty:
                all_comments.append(df)
        else:
            log.warning(f"  No comments file found for {video_id}")

    if not all_comments:
        log.error("No valid comments found across all videos. Exiting.")
        return

    # ── Merge ────────────────────────────────────────────────────────────────
    final_df = pd.concat(all_comments, ignore_index=True)
    final_df["transcript_context"] = final_df["video_id"].map(transcript_dict)

    # ── Enforce schema ───────────────────────────────────────────────────────
    schema = ["video_id", "comment_text", "clean_text", "sentiment_score",
              "sentiment_label", "comment_length", "word_count", "transcript_context"]
    for col in schema:
        if col not in final_df.columns:
            final_df[col] = ""
    final_df = final_df[schema]

    # ── Deduplicate ──────────────────────────────────────────────────────────
    before_dedup = len(final_df)
    final_df.drop_duplicates(subset=["video_id", "clean_text"], inplace=True)
    stats["comments_removed"] += before_dedup - len(final_df)

    # ── Save outputs ─────────────────────────────────────────────────────────
    out_final = base / "final_clean_dataset.csv"
    final_df.to_csv(out_final, index=False)
    log.info(f"Saved: {out_final}  ({len(final_df)} rows)")

    # Clean transcripts table
    transcript_df = pd.DataFrame([
        {"video_id": vid, "clean_transcript": txt}
        for vid, txt in transcript_dict.items()
    ])
    out_transcripts = base / "clean_transcripts.csv"
    transcript_df.to_csv(out_transcripts, index=False)
    log.info(f"Saved: {out_transcripts}")

    # ── Validation ───────────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  CLEANING REPORT")
    print("═" * 55)
    print(f"  Videos processed       : {stats['videos']}")
    print(f"  Videos without transcript: {stats['videos_no_transcript']}")
    print(f"  Comments kept          : {len(final_df)}")
    print(f"  Comments removed       : {stats['comments_removed']}")
    print(f"  Final dataset shape    : {final_df.shape}")
    print("═" * 55)
    print("\n  Sample rows:")
    print(final_df[["video_id", "clean_text", "sentiment_label"]].head(3).to_string(index=False))

    # Validate no garbage leaked through
    leak = final_df["clean_text"].str.contains(
        r"\b(video id|channel name|sequence)\b|\b\d+\s+\d+\s+\d+\s+\d+\s+\d+\b",
        regex=True, case=False, na=False
    )
    if leak.any():
        log.warning(f"  ⚠ {leak.sum()} rows may still contain artifacts — review manually.")
    else:
        print("\n  ✔ Validation passed — no artifacts detected.")
    print("═" * 55 + "\n")


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py /path/to/base_dir")
        sys.exit(1)
    run_pipeline(sys.argv[1])