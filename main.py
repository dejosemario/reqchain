import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"

load_dotenv(dotenv_path=BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

if not OPENROUTER_API_KEY:
    print("ERROR: OPENROUTER_API_KEY is not set. Add it to your .env file.", file=sys.stderr)
    sys.exit(1)

if not MODEL_NAME:
    print("ERROR: MODEL_NAME is not set. Add it to your .env file.", file=sys.stderr)
    sys.exit(1)


def load_prompt_text(filename: str) -> str:
    """Read a raw prompt template from the prompts/ directory."""
    path = PROMPTS_DIR / filename
    if not path.exists():
        print(f"ERROR: Prompt file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return path.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# LLM (OpenRouter, OpenAI-compatible) configuration
# ---------------------------------------------------------------------------

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.3,
)

output_parser = StrOutputParser()

