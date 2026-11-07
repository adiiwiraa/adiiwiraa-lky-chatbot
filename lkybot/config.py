"""Central configuration: env vars, paths, constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PDF_DIR = DATA_DIR / "pdfs"
DB_DIR = ROOT / "db"
RESULTS_DIR = ROOT / "results"
SCRAPE_LOG = DATA_DIR / "scrape_log.txt"
EVAL_SET = DATA_DIR / "eval_set.csv"


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ without overriding.

    Values already present in the process environment win; the .env file only
    fills in missing variables. Supports inline comments and quoted values.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value.startswith("#"):
            continue
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4.6")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")

COLLECTION_NAME = "lky_docs"
TOP_K = 5
CHUNK_MIN_WORDS = 200
CHUNK_MAX_WORDS = 350
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.35"))
