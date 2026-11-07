# WIiTek

[![checks](https://github.com/erybie222/wiitek/actions/workflows/checks.yml/badge.svg)](https://github.com/erybie222/wiitek/actions/workflows/checks.yml)

A chatbot that answers questions about studying at the Faculty of Computing and
Telecommunications (WIiT) of Poznań University of Technology: admission thresholds,
study plans (courses and ECTS points) and staff office hours.

The number one rule the bot is built around: **it never makes up a number.** Every figure in an
answer comes from the database together with its source and date. If the database has no
answer, the bot says it does not know. Hallucinations are a common problem with LLMs, and WIiTek is designed to avoid them.

## How it works

The project has two halves.

**Ingestion (offline).** Scrapers collect data from three official sources and store it in
PostgreSQL:

| Source                                           | What it provides                                             | Scraper                                         |
| ------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------- |
| ECTS course catalogs on put.poznan.pl            | courses per major and semester, ECTS, hours, exam or not     | `src/scrapers/html_ects.py`                     |
| Admission thresholds PDF                         | minimum points and number of places per major and year       | `src/scrapers/pdf_thresholds.py`                |
| Informator PP (the university's staff directory) | faculty units, staff, administrative functions, office hours | `src/scrapers/informator_staff_office_hours.py` |

**Answering (online).** An LLM agent (LangChain + OpenAI) receives the question and answers it
by calling read-only database tools — one tool per table (`src/database/repository.py`). It does
not search text by similarity: structured facts like thresholds or hours are always read with
SQL, because similarity search is unreliable on numbers. The agent's rules (cite the source, ask
when a name is ambiguous, refuse questions outside WIiT) are in its system prompt in
`src/templates.py`.

## Setup

You need [Docker](https://www.docker.com/) and [uv](https://github.com/astral-sh/uv), and
Python 3.12 or newer.

1. Install the dependencies:
   ```bash
   uv sync
   ```
2. Create your configuration file and fill it in:
   ```bash
   cp .env.example .env
   ```
   `.env.example` explains every setting. The database values can stay as they are. To chat with
   the bot or run the evaluation you need an `OPENAI_API_KEY`; ingestion works without one.
3. Start the database:
   ```bash
   docker-compose -f docker/docker-compose.yml --env-file .env up -d
   ```
   On the first start Docker creates the tables from `docker/init.sql`. That file only runs on an
   empty database: after changing it, either apply the change by hand or wipe the database (see
   below).

## Usage

### Load the data

```bash
uv run src/run_ingestion.py
```

This fills the database in a single transaction: majors and their ECTS courses, admission
thresholds, then faculty units and staff from Informator. Before running it:

- Put the thresholds PDF in `data/`, named `progi punktowe <anything>.pdf` (e.g.
  `progi punktowe 2025-2021.pdf`) — exactly one such file. The years are read from the table
  header, so a newer PDF just replaces the old file. Without it the thresholds step is skipped
  and everything else is still saved.
- For staff data, set `INFORMATOR_SESSION` in `.env` — a session cookie copied from your
  browser (`.env.example` says how). Informator protects its data with a captcha, so the scraper
  cannot log in by itself, and the cookie expires after a while. Without a valid cookie the staff
  step is skipped and everything else is still saved.

To start again from an empty database without recreating it:

```bash
uv run src/clean_db.py
```

### Chat with the bot

```bash
uv run main.py
```

Type a question in Polish or English; `exit` or Ctrl+C quits. The bot remembers the conversation
until you quit, so follow-ups like "a w 2024?" work; restart it for a fresh conversation. Before
each answer the chat prints which
database tools the agent called, which helps when an answer looks wrong.

### Measure answer quality

```bash
uv run python -m eval.run_eval                              # every case
uv run python -m eval.run_eval --category progi_punktowe    # one category
uv run python -m eval.run_eval --only prog-informatyka-2025 --verbose
uv run python -m eval.run_eval --no-judge                   # skip the LLM judges (faster, cheaper)
```

The evaluation asks the agent every question in `eval/benchmark.yaml` (hand-written questions with
verified answers) and scores each answer in two layers:

1. **Tool calls**, checked by code: did the agent call the right tools, and did they return the
   right data?
2. **The answer**, checked by a second LLM (the "judge"): is every claim backed by the data the
   agent received (_faithfulness_ — this is the hallucination check), and is the answer right
   compared to the reference (_correctness_)?

The console shows a summary; the full report, with every answer and tool call, is saved to
`eval/results/`, named by date and what was run (e.g. `2026-10-31_09-42-00_category-dopytania.json`).

The judge is a model too, so it is checked separately:

```bash
uv run python -m eval.calibrate_judges              # every calibration item once
uv run python -m eval.calibrate_judges --repeat 3   # judges are not deterministic
```

`eval/judge_calibration.yaml` holds answers whose right score is known — mostly broken on purpose
(an invented number, a wrong hour, a value from another row), plus correct ones that must not be
punished. The command reports how many broken answers the judges caught and whether any correct
answer was scored 0. No database is needed.

```bash
uv run python -m eval.check_leakage            # no benchmark value in any prompt
uv run python -m eval.check_prompt_formatting  # the judges' prompts fill in correctly
```

`check_leakage` checks that no prompt, tool description or judge schema contains a value or
question from the benchmark or the calibration set — a prompt holding the answers would turn the
evaluation into a memory test. It exits with an error and lists every match.
`check_prompt_formatting` builds the judges' prompts without calling a model, to catch a template
that breaks, or one that silently leaves out an input such as the answer being judged.

Neither needs a database or an API key, so both run in CI on every push
(`.github/workflows/checks.yml`, status badge at the top), together with a check that `uv.lock`
is up to date and that every module imports.

## Managing the database

**pgAdmin** (a web interface for browsing the database) runs at `http://localhost:5050`. Log in
with `PGADMIN_EMAIL` / `PGADMIN_PASSWORD` from `.env`, then add a server:

- **Host name/address:** `PUT_Bot_db` (the container name — pgAdmin reaches the database inside
  Docker's network, so `localhost` does not work here)
- **Port:** `5432`
- **Maintenance database / Username / Password:** `DB_NAME` / `DB_USER` / `DB_PASSWORD` from `.env`

Stop the containers (the data is kept):

```bash
docker-compose -f docker/docker-compose.yml --env-file .env down
```

Stop them **and delete all data** (needed after changing `docker/init.sql`):

```bash
docker-compose -f docker/docker-compose.yml --env-file .env down -v
```

## Project layout

```
main.py                  terminal chat
src/
  run_ingestion.py       loads all data into the database
  clean_db.py            deletes all loaded data, keeps the tables
  agent.py               builds the agent and asks it a question
  templates.py           all prompts: the agent's and the judges'
  config.py              settings read from .env
  scrapers/              one scraper per data source
  database/
    queries/             all SQL, one file per table
    repository.py        the database tools the agent can call
    connection.py        shared database connection pool
eval/
  benchmark.yaml         evaluation questions with verified answers
  run_eval.py            runs the evaluation (start here)
  tool_call_scoring.py   layer 1: tool calls
  answer_judges.py       layer 2: LLM judges
  report.py              summary and JSON report
docker/
  docker-compose.yml     database + pgAdmin
  init.sql               table definitions
.github/workflows/
  checks.yml             CI: the checks that need no database or API key
```
