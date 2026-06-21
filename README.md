# Intelligent Software Requirements Analysis System

A LangChain LCEL prompt chain that takes a client's free-text project
description and produces a structured initial project assessment, by
running the request through five sequential reasoning stages:

1. **Interpret the Project Request** — understand the client's description and main business objective.
2. **Identify Possible Project Categories** — suggest candidate categories.
3. **Select the Best Category** — choose the single most appropriate category.
4. **Extract Missing Requirements** — flag information needed before implementation can begin.
5. **Generate an Initial Assessment** — produce a concise handoff summary.

Each stage's output is piped into the next stage's input using LangChain
Expression Language (LCEL) composition (`prompt | llm | output_parser`,
threaded together with `RunnableLambda` steps).

## Project Structure

```
requirements-analyzer/
├── main.py                              # Entry point, builds and runs the LCEL chain
├── prompts/
│   ├── 01_interpret_request.txt
│   ├── 02_identify_categories.txt
│   ├── 03_select_best_category.txt
│   ├── 04_extract_missing_requirements.txt
│   └── 05_generate_assessment.txt
├── requirements.txt
├── .env.example                         # Template — copy to .env and fill in
└── .gitignore
```

## Setup

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd requirements-analyzer
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Copy the example file and fill in your own values:

   ```bash
   cp .env.example .env
   ```

   Edit `.env`:

   ```
   OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxx
   MODEL_NAME=openai/gpt-4o-mini
   ```

   Get an OpenRouter API key at https://openrouter.ai/keys. `MODEL_NAME`
   can be any model slug OpenRouter supports (e.g. `openai/gpt-4o-mini`,
   `anthropic/claude-3.5-sonnet`, `meta-llama/llama-3.1-70b-instruct`).

   The `.env` file is git-ignored and must never be committed.

## Usage

Run the script with the client's project description as the first
command-line argument:

```bash
python main.py "We need a system for our customers to order food online and track delivery in real time. We also want restaurant partners to manage their own menus."
```

The script prints the output of each of the five stages as it completes,
then prints the final assessment at the end under a clearly marked banner.

## How the Chain Is Composed (LCEL)

Each stage is its own LCEL pipeline:

```python
interpret_chain = (
    interpret_prompt
    | llm
    | output_parser
    | make_printer("1. Interpret the Project Request")
)
```

The five stage-chains are then threaded together through `RunnableLambda`
steps that pass an accumulating dictionary of state from one stage to the
next, and the whole pipeline is itself a single composed `Runnable`:

```python
full_chain = (
    RunnableLambda(run_stage_1)
    | RunnableLambda(run_stage_2)
    | RunnableLambda(run_stage_3)
    | RunnableLambda(run_stage_4)
    | RunnableLambda(run_stage_5)
)
```

This keeps every individual stage as a clean, reusable LCEL chain
(`prompt | model | parser`) while still composing the full five-stage
workflow as one runnable pipeline, satisfying the
Understanding → Classification → Validation → Requirement Extraction →
Assessment flow.

## Categories

The system classifies every request into exactly one of:

- Web Application
- Mobile Application
- API / Backend Service
- Data Analytics Platform
- AI / Machine Learning System
- E-Commerce Platform
- Enterprise Management System
- System Integration
- DevOps / Infrastructure Automation
- General Software Project

## Notes

- No API key is hardcoded anywhere in the source — both `OPENROUTER_API_KEY`
  and `MODEL_NAME` are loaded from `.env` via `python-dotenv`.
- The LLM is accessed through `langchain-openai`'s `ChatOpenAI` client
  pointed at OpenRouter's OpenAI-compatible endpoint
  (`https://openrouter.ai/api/v1`), so any OpenRouter-hosted model can be
  used by changing `MODEL_NAME`.
