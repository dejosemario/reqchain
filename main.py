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

def make_printer(stage_title: str):
    """Returns a RunnableLambda that prints text passing through the chain."""

    def _print_and_pass(text: str) -> str:
        print("\n" + "=" * 80)
        print(f"STAGE: {stage_title}")
        print("=" * 80)
        print(text.strip())
        return text

    return RunnableLambda(_print_and_pass)


# ---------------------------------------------------------------------------
# Stage 1: Interpret the Project Request
# ---------------------------------------------------------------------------

interpret_prompt = PromptTemplate.from_template(load_prompt_text("01_interpret_request.txt"))
interpret_chain = (
    interpret_prompt
    | llm
    | output_parser
    | make_printer("1. Interpret the Project Request")
)

# ---------------------------------------------------------------------------
# Stage 2: Identify Possible Project Categories
# ---------------------------------------------------------------------------

identify_categories_prompt = PromptTemplate.from_template(
    load_prompt_text("02_identify_categories.txt")
)
identify_categories_chain = (
    identify_categories_prompt
    | llm
    | output_parser
    | make_printer("2. Identify Possible Project Categories")
)

# ---------------------------------------------------------------------------
# Stage 3: Select the Best Category
# ---------------------------------------------------------------------------

select_category_prompt = PromptTemplate.from_template(
    load_prompt_text("03_select_best_category.txt")
)
select_category_chain = (
    select_category_prompt
    | llm
    | output_parser
    | make_printer("3. Select the Best Category")
)

# ---------------------------------------------------------------------------
# Stage 4: Extract Missing Requirements
# ---------------------------------------------------------------------------

extract_missing_prompt = PromptTemplate.from_template(
    load_prompt_text("04_extract_missing_requirements.txt")
)
extract_missing_chain = (
    extract_missing_prompt
    | llm
    | output_parser
    | make_printer("4. Extract Missing Requirements")
)

# ---------------------------------------------------------------------------
# Stage 5: Generate an Initial Assessment
# ---------------------------------------------------------------------------

generate_assessment_prompt = PromptTemplate.from_template(
    load_prompt_text("05_generate_assessment.txt")
)
generate_assessment_chain = (
    generate_assessment_prompt
    | llm
    | output_parser
)

def extract_category_name(selection_text: str) -> str:
    """Extracts clean category names by parsing out prefix headers."""
    for line in selection_text.splitlines():
        if line.strip().lower().startswith("selected category:"):
            return line.split(":", 1)[1].strip()
    return selection_text.strip()

# ---------------------------------------------------------------------------
# Full pipeline composed with LCEL
# ---------------------------------------------------------------------------
#
# Each stage's output is threaded into the next stage's input dict using
# RunnableLambda steps, forming a single LCEL-composed pipeline:
#
#   client_request
#       -> interpretation
#       -> categories
#       -> selected_category (+ justification)
#       -> missing_requirements
#       -> final_assessment

def build_full_chain():
    def run_stage_1(inputs: dict) -> dict:
        interpretation = interpret_chain.invoke({"client_request": inputs["client_request"]})
        return {**inputs, "interpretation": interpretation}
    
    def run_stage_2(inputs: dict) -> dict:
        categories = identify_categories_chain.invoke({"interpretation": inputs["interpretation"]})
        return {**inputs, "categories": categories}

    def run_stage_3(inputs: dict) -> dict:
        selection_text = select_category_chain.invoke(
            {
                "interpretation": inputs["interpretation"],
                "categories": inputs["categories"],
            }
        )
        selected_category = extract_category_name(selection_text)
        return {**inputs, "selection_text": selection_text, "selected_category": selected_category}
    
    def run_stage_4(inputs: dict) -> dict:
        missing_requirements = extract_missing_chain.invoke(
            {
                "interpretation": inputs["interpretation"],
                "selected_category": inputs["selected_category"],
                "client_request": inputs["client_request"],
            }
        )
        return {**inputs, "missing_requirements": missing_requirements}

    def run_stage_5(inputs: dict) -> dict:
        final_assessment = generate_assessment_chain.invoke(
            {
                "selected_category": inputs["selected_category"],
                "interpretation": inputs["interpretation"],
                "missing_requirements": inputs["missing_requirements"],
            }
        )
        return {**inputs, "final_assessment": final_assessment}
    
    full_chain = (
        RunnableLambda(run_stage_1)
        | RunnableLambda(run_stage_2)
        | RunnableLambda(run_stage_3)
        | RunnableLambda(run_stage_4)
        | RunnableLambda(run_stage_5)
    )
    return full_chain

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py \"<client project description>\"",file=sys.stderr)
        sys.exit(1)
        
    client_request = sys.argv[1]
    
    chain = build_full_chain()
    result  = chain.invoke({"client_request": client_request})
    
    print("\n" + "#" * 80)
    print("# FINAL PROJECT ASSESSMENT")
    print("#" * 80)
    print(result["final_assessment"].strip())
    print()

if __name__ == "__main__":
    main()
    
