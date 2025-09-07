import os
import logging
from typing import Optional, Tuple, List

import pandas as pd
from pandas.errors import ParserError


logger = logging.getLogger(__name__)


def _safe_read_csv(file_path: str, encodings: Tuple[str, ...] = ("utf-8", "latin-1", "cp1252")) -> pd.DataFrame:
    """Attempt to read a CSV using multiple encodings.

    Raises exceptions up to the caller except UnicodeDecodeError, which will
    trigger a fallback to the next encoding in the list.
    """
    last_decode_error: Optional[Exception] = None
    for enc in encodings:
        try:
            logger.debug(f"Trying to read CSV with encoding='{enc}'")
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError as ude:
            last_decode_error = ude
            logger.warning(f"Unicode decode error with encoding '{enc}', trying next fallback if available.")
            continue
        except ParserError as pe:
            logger.warning(
                f"Parser error with C engine for encoding '{enc}': {pe}. "
                "Falling back to python engine with on_bad_lines='skip'."
            )
            try:
                # More tolerant parser that can skip malformed rows
                return pd.read_csv(
                    file_path,
                    encoding=enc,
                    engine="python",
                    on_bad_lines="skip",
                )
            except Exception as e2:
                logger.warning(f"Fallback python engine also failed for encoding '{enc}': {e2}")
                continue
        except Exception as e:
            # Generic error (including tokenizing errors surfaced as ValueError)
            if isinstance(e, ValueError) and "Expected" in str(e) and "saw" in str(e):
                logger.warning(
                    f"Tokenizing error with C engine for encoding '{enc}': {e}. "
                    "Falling back to python engine with on_bad_lines='skip'."
                )
                try:
                    return pd.read_csv(
                        file_path,
                        encoding=enc,
                        engine="python",
                        on_bad_lines="skip",
                    )
                except Exception as e2:
                    logger.warning(f"Fallback python engine also failed for encoding '{enc}': {e2}")
                    continue
            raise
    # If we exhausted encodings, re-raise a meaningful error
    message = (
        f"Failed to decode CSV '{file_path}' using encodings {list(encodings)}. "
        f"Last error: {last_decode_error}"
    )
    raise UnicodeDecodeError("csv", b"", 0, 1, message)


# Load the dataset
def load_dataset(file_path: str) -> Optional[pd.DataFrame]:
    """Load dataset from a CSV file path with validation and clear errors.

    - Uses the provided file_path (no hardcoding)
    - Validates file existence and non-empty content
    - Validates required columns: 'class' and 'message'
    """
    if not file_path or not isinstance(file_path, str):
        logger.error("load_dataset: 'file_path' must be a non-empty string.")
        return None

    if not os.path.exists(file_path):
        logger.error(f"load_dataset: File not found at path: {file_path}")
        return None

    try:
        df = _safe_read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"load_dataset: File not found at path: {file_path}")
        return None
    except Exception as e:
        logger.error(f"load_dataset: Error reading CSV '{file_path}': {e}")
        return None

    if df is None or df.empty:
        logger.error("load_dataset: Loaded DataFrame is empty.")
        return None

    # Normalize and sanitize column names
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Drop unnamed/empty columns that sometimes appear in Kaggle variants
    df = df[[c for c in df.columns if not c.startswith("unnamed")]]

    # Map common schema variants to expected names
    if {"class", "message"}.issubset(df.columns):
        df = df[["class", "message"]].copy()
    elif {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].copy()
        df.columns = ["class", "message"]
    elif {"label", "message"}.issubset(df.columns):
        df = df[["label", "message"]].copy()
        df.columns = ["class", "message"]
    else:
        logger.error(
            "load_dataset: CSV does not contain recognizable columns for labels and messages. "
            f"Found columns: {list(df.columns)}. Expected one of: ('class','message') or ('v1','v2')."
        )
        return None

    return df


def load_and_merge_datasets(file_paths: List[str]) -> Optional[pd.DataFrame]:
    """Load and merge multiple datasets, resolving common filename variants.

    - Accepts names with or without the .csv extension
    - Skips files that do not exist
    - Uses load_dataset for robust parsing and schema normalization
    - De-duplicates rows by normalized message content
    """
    if not file_paths:
        logger.error("load_and_merge_datasets: No file paths provided.")
        return None

    def resolve_candidates(path_like: str) -> List[str]:
        candidates = []
        if not path_like:
            return candidates
        base, ext = os.path.splitext(path_like)
        if ext:
            candidates.append(path_like)
        else:
            candidates.extend([path_like, f"{path_like}.csv"])  # try bare and with .csv
        # Also try within a conventional data directory
        candidates.extend([
            os.path.join("data", os.path.basename(path_like)),
            os.path.join("data", f"{os.path.basename(base)}.csv"),
        ])
        return candidates

    resolved: List[str] = []
    for p in file_paths:
        for candidate in resolve_candidates(p):
            if os.path.exists(candidate):
                resolved.append(candidate)
                break
        else:
            logger.warning(f"load_and_merge_datasets: Dataset not found: {p}")

    if not resolved:
        logger.error("load_and_merge_datasets: None of the provided datasets were found.")
        return None

    frames: List[pd.DataFrame] = []
    for path in resolved:
        df = load_dataset(path)
        if df is None or df.empty:
            logger.warning(f"load_and_merge_datasets: Skipping empty/invalid dataset: {path}")
            continue
        frames.append(df)

    if not frames:
        logger.error("load_and_merge_datasets: No valid data loaded from provided files.")
        return None

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    # Normalize message for deduplication across datasets
    merged["__message_norm__"] = merged["message"].astype(str).str.strip().str.lower()
    merged = merged.drop_duplicates(subset="__message_norm__", keep="first").drop(columns=["__message_norm__"])
    dropped = before - len(merged)
    if dropped > 0:
        logger.info(f"load_and_merge_datasets: Dropped {dropped} duplicate messages across datasets.")

    logger.info(f"load_and_merge_datasets: Loaded {len(merged)} total rows from {len(frames)} datasets.")
    return merged


# Data preprocessing
def preprocess_data(data: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Clean and validate the dataset, and map labels to numeric.

    Steps:
    - Validate required columns exist ('class' or already 'label', and 'message')
    - Normalize/rename to columns: 'label' (int 0/1) and 'message' (str)
    - Strip whitespace, drop empty or null messages
    - Drop rows with malformed labels
    - Log basic stats (counts of ham/spam)
    """
    if data is None:
        logger.error("preprocess_data: Received None instead of a DataFrame.")
        return None

    if not isinstance(data, pd.DataFrame) or data.empty:
        logger.error("preprocess_data: Input DataFrame is invalid or empty.")
        return None

    df = data.copy()

    # Ensure we have a message column
    if "message" not in df.columns:
        logger.error("preprocess_data: Missing required column 'message'.")
        return None

    # Accept either 'class' (expected) or pre-existing 'label'
    if "label" in df.columns:
        label_col = "label"
    elif "class" in df.columns:
        label_col = "class"
    else:
        logger.error("preprocess_data: Missing required label column ('class' or 'label').")
        return None

    # Reduce to relevant columns and standardize names
    df = df[[label_col, "message"]].copy()
    df.columns = ["label", "message"]

    # Normalize message text: ensure string, strip, and drop empties
    df["message"] = df["message"].astype(str).str.strip()
    before_drop_empty = len(df)
    df = df[df["message"].notna() & (df["message"].str.len() > 0)]
    dropped_empty = before_drop_empty - len(df)
    if dropped_empty > 0:
        logger.info(f"preprocess_data: Dropped {dropped_empty} rows with empty messages.")

    # Normalize labels then map to numeric 0/1
    # Accept a few common variations
    label_map = {"ham": 0, "spam": 1, "0": 0, "1": 1, 0: 0, 1: 1}
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map(label_map)

    before_drop_malformed = len(df)
    df = df[df["label"].isin([0, 1])]
    dropped_malformed = before_drop_malformed - len(df)
    if dropped_malformed > 0:
        logger.info(f"preprocess_data: Dropped {dropped_malformed} rows with malformed labels.")

    if df.empty:
        logger.error("preprocess_data: No usable data after cleaning.")
        return None

    # Compute and log basic stats
    total = len(df)
    ham = int((df["label"] == 0).sum())
    spam = int((df["label"] == 1).sum())
    logger.info(
        f"preprocess_data: Loaded {total} rows after cleaning | ham={ham} | spam={spam}"
    )

    return df


# Vectorize the text data
def vectorize_text(
    train_data,
    test_data,
    stop_words: str = "english",
    max_df: float = 0.7,
    min_df: int = 1,
    ngram_range: Tuple[int, int] = (1, 1),
):
    """Fit a TF-IDF vectorizer on train data and transform both sets.

    Returns (vectorizer, train_vec, test_vec). Returns (None, None, None) if inputs
    are invalid, logging the error.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if train_data is None or test_data is None:
        logger.error("vectorize_text: train_data and test_data must not be None.")
        return None, None, None

    if len(train_data) == 0 or len(test_data) == 0:
        logger.error("vectorize_text: train_data and test_data must be non-empty.")
        return None, None, None

    try:
        vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            max_df=max_df,
            min_df=min_df,
            ngram_range=ngram_range,
        )
        train_vec = vectorizer.fit_transform(train_data)
        test_vec = vectorizer.transform(test_data)
        return vectorizer, train_vec, test_vec
    except Exception as e:
        logger.error(f"vectorize_text: Error during vectorization: {e}")
        return None, None, None
