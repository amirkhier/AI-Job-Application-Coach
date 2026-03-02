# Phase 5: Evaluation, Optimization & Final Documentation

## Implementation Plan

**Version:** 1.0  
**Date:** March 2026  
**Author:** Senior Engineering  
**Status:** Complete  
**Estimated Duration:** 4–6 weeks  

---

## Table of Contents

1. [Overview](#overview)  
2. [Phase 5 Scope](#phase-5-scope)  
3. [Step-by-Step Implementation](#step-by-step-implementation)  
4. [Evaluation & Metrics](#evaluation--metrics)  
5. [A/B Testing Strategy](#ab-testing-strategy)  
6. [Documentation Plan](#documentation-plan)  
7. [Risks & Assumptions](#risks--assumptions)  

---

## Overview

Phase 5 is the capstone phase of the AI Job Application Coach project. With Phases 1–4 delivering a fully functional multi-agent system (6 LLM-powered agents, LangGraph orchestration, RAG pipeline, async task processing, security middleware, and Docker deployment), Phase 5 shifts focus to **measurable quality**, **prompt engineering excellence**, **data-driven optimization**, and **production-grade documentation**.

### Technical Objectives

| # | Objective | Success Criteria |
|---|-----------|-----------------|
| 1 | **Evaluation Framework** | Automated scoring pipeline covering all 6 agents with reproducible benchmarks |
| 2 | **Prompt Optimization** | ≥15% improvement in agent output quality scores vs. baseline |
| 3 | **A/B Testing Infrastructure** | Feature-flag-driven variant routing with statistical significance testing |
| 4 | **Error Handling Refinement** | Zero unhandled exceptions; all 8 known test failures resolved |
| 5 | **Final Documentation** | Complete API reference, architecture guide, runbook, and demo materials |
| 6 | **Performance Benchmarking** | Latency baselines for all endpoints; p95 < 5s for single-agent, < 10s for graph |

---

## Phase 5 Scope

### In Scope

- Evaluation framework with automated scoring for all agents (resume, interview, knowledge, job_search, memory, router)
- Prompt versioning, optimization, and regression testing
- A/B testing infrastructure leveraging the existing `USE_LANGGRAPH` feature flag pattern
- Comprehensive metrics collection and KPI dashboards
- Resolution of 8 known test failures (4 in `test_async_tasks.py`, 4 in `test_application_tracker.py`)
- Final documentation suite: API reference, architecture diagrams, deployment runbook, user guide
- Demo preparation and walkthrough materials

### Out of Scope

- New agent development or new endpoint creation
- Infrastructure changes beyond what is defined in the existing `docker-compose.yml`
- Production deployment execution (covered in `PHASE4_DEPLOYMENT_ROADMAP.md`)
- UI/frontend development

---

## Step-by-Step Implementation

### Milestone 1: Evaluation Framework (Week 1–2)

#### Step 1.1 — Define Evaluation Dataset

**Objective:** Create a curated, version-controlled test corpus for each agent.

**Actions:**
1. Create `evaluation/datasets/` directory with subdirectories per agent:
   - `evaluation/datasets/resume/` — 20+ sample resumes (varied quality, industries, experience levels)
   - `evaluation/datasets/interview/` — 30+ question-answer pairs across roles and difficulty levels
   - `evaluation/datasets/knowledge/` — 40+ career questions with expected answer coverage
   - `evaluation/datasets/job_search/` — 15+ location/role search queries with expected results
   - `evaluation/datasets/router/` — 50+ user messages with ground-truth intent labels
2. Define a JSON schema for each dataset entry (input, expected output dimensions, human-annotated scores).
3. Include edge cases: empty inputs, adversarial inputs, multilingual queries, ambiguous intents.

**Deliverable:** `evaluation/datasets/` with `README.md` describing the schema and dataset composition.

#### Step 1.2 — Build Scoring Pipeline

**Objective:** Automated evaluation harness that scores agent outputs against defined criteria.

**Actions:**
1. Create `evaluation/scoring.py` — core scoring engine:
   ```
   Scoring Dimensions per Agent:
   ├── Resume Agent
   │   ├── Accuracy: Are identified strengths/weaknesses valid?
   │   ├── Completeness: Are all resume sections covered?
   │   ├── Actionability: Are recommendations specific and implementable?
   │   └── ATS Relevance: Does keyword analysis match job description?
   ├── Interview Agent
   │   ├── Question Relevance: Are questions appropriate for role/level?
   │   ├── Evaluation Fairness: Are scores consistent across similar answers?
   │   ├── Feedback Quality: Is feedback specific and constructive?
   │   └── Session Summary Accuracy: Does summary reflect actual performance?
   ├── Knowledge Agent
   │   ├── Factual Accuracy: Are answers grounded in RAG sources?
   │   ├── Source Attribution: Are citations present and correct?
   │   ├── Confidence Calibration: Does confidence score match actual accuracy?
   │   └── Completeness: Are all aspects of the question addressed?
   ├── Job Search Agent
   │   ├── Location Accuracy: Are geocoded results correct?
   │   ├── Listing Relevance: Do jobs match the search criteria?
   │   ├── Match Scoring: Are profile-score calculations reasonable?
   │   └── Fallback Quality: Are fallback listings useful when APIs fail?
   ├── Router Agent
   │   ├── Classification Accuracy: Intent matches ground truth?
   │   ├── Confidence Calibration: Threshold behavior is correct?
   │   ├── Session Override: Active interview sessions are respected?
   │   └── Fallback Consistency: Keyword fallback aligns with LLM classification?
   └── Memory Agent
       ├── Context Recall: Is loaded context relevant and complete?
       ├── Summary Quality: Are conversation summaries accurate?
       ├── Profile Extraction: Are extracted profile fields correct?
       └── Merge Correctness: Does profile merging preserve data?
   ```
2. Create `evaluation/evaluator.py` — orchestrator that runs all agents against datasets and collects scores.
3. Create `evaluation/reporters.py` — generates JSON and Markdown reports with score distributions, histograms, and comparison tables.

**Tools & Libraries:**
- `pytest` + `pytest-benchmark` — test execution and timing
- `numpy` / `statistics` — score aggregation and statistical analysis
- `json` — structured report output
- `deepeval` (optional) — LLM-as-judge evaluation framework for subjective quality scoring
- `ragas` (optional) — RAG-specific evaluation metrics (faithfulness, answer relevancy, context precision)

**Deliverable:** `evaluation/scoring.py`, `evaluation/evaluator.py`, `evaluation/reporters.py`

#### Step 1.3 — Establish Baselines

**Objective:** Record current agent performance as the baseline for all future optimization.

**Actions:**
1. Run the evaluation pipeline against all datasets with the current prompts.
2. Record scores in `evaluation/baselines/baseline_v1.json`.
3. Document baseline scores in a summary table:
   | Agent | Accuracy | Completeness | Actionability | Overall |
   |-------|----------|-------------|---------------|---------|
   | Resume | — | — | — | — |
   | Interview | — | — | — | — |
   | Knowledge | — | — | — | — |
   | Job Search | — | — | — | — |
   | Router | — | — | — | — |
   | Memory | — | — | — | — |
4. Identify the lowest-performing areas to prioritize in prompt optimization.

**Deliverable:** `evaluation/baselines/baseline_v1.json`, baseline summary in evaluation README.

---

### Milestone 2: Prompt Optimization (Week 2–3)

#### Step 2.1 — Prompt Inventory & Versioning

**Objective:** Catalog all prompts, establish version control, and enable rollback.

**Actions:**
1. Create `evaluation/prompts/` directory with one file per agent:
   - `resume_prompts.py` — prompts from `app/agents/resume.py` (`analyze_resume`, `suggest_improvements`)
   - `interview_prompts.py` — prompts from `app/agents/interview.py` (`generate_questions`, `evaluate_answer`, `generate_session_summary`)
   - `knowledge_prompts.py` — prompts from `app/agents/knowledge.py` (`answer_question`, `get_topic_summary`)
   - `memory_prompts.py` — prompts from `app/agents/memory.py` (`summarize`, `extract_profile`, `generate_insights`)
   - `job_search_prompts.py` — prompts from `app/agents/job_search.py` (`search_jobs`, `search_jobs_with_matching`)
   - `router_prompts.py` — prompts from `app/agents/router.py` (`classify_intent`)
   - `summary_prompts.py` — prompts from `app/graph/workflow.py` (`_summary_agent`)
2. Each prompt file exports a dict of `{version: str, prompt_name: str, template: str, variables: list}`.
3. Implement `evaluation/prompt_manager.py` — loads prompt variants by version string, supports A/B variant selection.

**Deliverable:** `evaluation/prompts/` directory, `evaluation/prompt_manager.py`

#### Step 2.2 — Systematic Prompt Improvement

**Objective:** Iteratively improve prompts using evaluation data.

**Actions:**
1. Apply prompt engineering best practices to each agent's prompts:
   - **Structured output specification:** Reinforce JSON schema requirements with explicit field descriptions and examples.
   - **Few-shot examples:** Add 1–2 exemplar input/output pairs where quality scores are lowest.
   - **Chain-of-thought:** For complex scoring tasks (resume analysis, answer evaluation), add `"Think step by step"` reasoning guidance.
   - **Negative examples:** For router classification, include examples of ambiguous inputs with correct classifications.
   - **Context window optimization:** Trim unnecessary instructions; prioritize signal-dense context for RAG prompts.
2. For each prompt revision:
   - Increment the version string (e.g., `v1.0` → `v1.1`).
   - Run the evaluation pipeline against the same dataset.
   - Record scores in `evaluation/baselines/optimized_v{X}.json`.
   - Accept the revision only if overall score improves without regression in any dimension.
3. Target ≥15% improvement in the lowest-scoring dimensions.

**Iteration Protocol:**
```
For each agent:
  1. Review baseline scores → identify weakest dimensions
  2. Draft prompt revision targeting weakest dimensions
  3. Run evaluation against dataset
  4. Compare scores to baseline
  5. If improved: accept, tag version, record results
  6. If regressed: analyze failure cases, revise, re-evaluate
  7. Maximum 3 iterations per prompt before accepting best version
```

**Deliverable:** Optimized prompt versions with documented score improvements.

#### Step 2.3 — Prompt Regression Testing

**Objective:** Ensure prompt changes do not degrade agent behavior.

**Actions:**
1. Create `tests/test_prompt_regression.py`:
   - For each agent, define 5–10 "golden" input/output pairs that must always pass.
   - Test that optimized prompts still produce outputs matching these golden cases within an acceptable tolerance.
2. Integrate prompt regression tests into the existing pytest suite.
3. Add a CI-compatible script `evaluation/run_regression.py` that exits non-zero on any regression.

**Deliverable:** `tests/test_prompt_regression.py`, `evaluation/run_regression.py`

---

### Milestone 3: A/B Testing Infrastructure (Week 3–4)

#### Step 3.1 — Feature Flag Extension

**Objective:** Extend the existing `USE_LANGGRAPH` flag pattern to support multi-variant experiments.

**Actions:**
1. Create `evaluation/ab_testing.py`:
   ```python
   # Core A/B testing module
   class ABTestManager:
       """Manages experiment assignment and tracking."""
       - register_experiment(name, variants, traffic_split)
       - get_variant(experiment_name, user_id) -> str
       - record_outcome(experiment_name, user_id, variant, metrics)
       - get_results(experiment_name) -> ExperimentResults
   ```
2. Variant assignment strategy:
   - **Deterministic hashing:** `hash(user_id + experiment_name) % 100` — ensures consistent variant per user.
   - **Traffic splitting:** Configurable percentage per variant (e.g., 50/50, 80/20 for canary releases).
   - **Override support:** Environment variable or config-based force-assignment for testing.
3. Integrate with the existing `Settings` class in `app/config.py` for experiment configuration.

**Deliverable:** `evaluation/ab_testing.py`

#### Step 3.2 — Experiment Definitions

**Objective:** Define concrete A/B experiments for Phase 5 validation.

**Experiments:**

| # | Experiment | Variants | Primary Metric | Duration |
|---|-----------|----------|---------------|----------|
| 1 | `resume_prompt_v2` | A: current prompt, B: optimized prompt | Resume analysis overall score | 1 week |
| 2 | `interview_cot` | A: direct scoring, B: chain-of-thought scoring | Evaluation fairness score | 1 week |
| 3 | `router_confidence_threshold` | A: 0.7 threshold, B: 0.6 threshold, C: 0.8 threshold | Classification accuracy | 1 week |
| 4 | `rag_chunk_size` | A: 800 tokens, B: 600 tokens, C: 1000 tokens | Knowledge answer accuracy + latency | 1 week |
| 5 | `graph_vs_direct` | A: USE_LANGGRAPH=true, B: USE_LANGGRAPH=false | End-to-end response quality + latency | 1 week |

**Actions:**
1. Create `evaluation/experiments/` with one config file per experiment.
2. Each config defines: `experiment_name`, `variants`, `traffic_split`, `primary_metric`, `secondary_metrics`, `min_sample_size`, `significance_level`.

**Deliverable:** `evaluation/experiments/` directory with experiment configs.

#### Step 3.3 — Statistical Analysis

**Objective:** Implement rigorous statistical testing for experiment results.

**Actions:**
1. Create `evaluation/statistics.py`:
   - **Sample size calculator:** Given baseline rate, minimum detectable effect, and significance level, compute required sample size.
   - **Significance testing:** Two-sample t-test for continuous metrics, chi-squared test for categorical outcomes.
   - **Confidence intervals:** 95% CI for all reported metrics.
   - **Multiple comparison correction:** Bonferroni or Benjamini-Hochberg for experiments with >2 variants.
2. Decision framework:
   ```
   If p-value < 0.05 AND effect_size > minimum_detectable_effect:
       → Adopt winning variant
   If p-value < 0.05 AND effect_size < minimum_detectable_effect:
       → Result is statistically significant but practically insignificant; keep current
   If p-value >= 0.05:
       → Insufficient evidence; extend experiment or keep current
   ```

**Tools & Libraries:**
- `scipy.stats` — t-test, chi-squared, confidence intervals
- `numpy` — numerical computations
- `statistics` (stdlib) — basic descriptive statistics

**Deliverable:** `evaluation/statistics.py`

#### Step 3.4 — Results Collection & Reporting

**Objective:** Automated experiment result collection and decision reports.

**Actions:**
1. Create `evaluation/experiment_reporter.py`:
   - Collects all experiment outcomes from the `ABTestManager`.
   - Runs statistical analysis per experiment.
   - Generates a Markdown report per experiment with: summary table, metric distributions per variant, statistical test results, and recommendation (adopt/reject/extend).
2. Store experiment results in `evaluation/results/` with timestamped filenames.

**Deliverable:** `evaluation/experiment_reporter.py`, `evaluation/results/` directory.

---

### Milestone 4: Test Suite Hardening & Error Handling (Week 4)

#### Step 4.1 — Fix Known Test Failures

**Objective:** Resolve all 8 known test failures to achieve a clean test suite.

**Failures to Fix:**

| File | Count | Root Cause | Fix Strategy |
|------|-------|-----------|--------------|
| `test_async_tasks.py` | 4 | `@patch` target paths reference `DatabaseManager` from task modules where it is not directly importable | Correct the patch paths to target the actual import location (e.g., `app.tools.database.DatabaseManager`) |
| `test_application_tracker.py` | 4 | `ApplicationUpdateRequest` Pydantic model with `Optional` fields does not accept partial JSON in test payloads | Adjust test payloads to match the model's expected schema, or make model fields truly Optional with defaults of `None` |

**Actions:**
1. Diagnose each failure by running the specific test file with `-v --tb=long`.
2. Apply minimal, targeted fixes.
3. Verify all 83+ tests pass with `pytest --tb=short -q`.
4. Add a CI validation script: `scripts/run_tests.sh` / `scripts/run_tests.bat`.

**Deliverable:** All tests passing; CI validation script.

#### Step 4.2 — Error Handling Audit

**Objective:** Ensure zero unhandled exceptions across all code paths.

**Actions:**
1. Audit all agent modules for error paths:
   - `_parse_llm_json()` in `resume.py` — verify all JSON parse failures are caught.
   - `_error_analysis()` fallback in `resume.py` — confirm it returns valid structure.
   - `generate_questions()` default fallback in `interview.py` — ensure 5 default questions are always valid.
   - `_fallback_answer()` in `knowledge.py` — verify it runs when RAG or LLM fails.
   - `search_jobs()` fallback in `job_search.py` — confirm geocoding/Overpass failures produce useful fallback listings.
   - `classify_intent()` keyword fallback in `router.py` — verify confidence threshold behavior.
2. Audit middleware error handling:
   - `InputValidationMiddleware` — confirm SQL injection detection rejects cleanly without leaking stack traces.
   - `RateLimitMiddleware` — verify 429 responses include `Retry-After` header.
   - `ApiKeyMiddleware` — confirm 401 responses are generic (no key leakage).
3. Audit database error handling:
   - `DatabaseManager.ensure_connection()` — verify reconnection logic under connection pool exhaustion.
   - `DatabaseTransaction` — confirm rollback on exception.
4. Add structured error logging with correlation IDs for all caught exceptions (leveraging existing `RequestIdMiddleware`).

**Deliverable:** Error handling audit report; any critical fixes applied.

#### Step 4.3 — Performance Benchmarking

**Objective:** Establish latency baselines for all endpoints.

**Actions:**
1. Create `evaluation/benchmarks.py`:
   - Use `pytest-benchmark` or a custom timing harness.
   - Benchmark each endpoint with mocked LLM responses to isolate infrastructure latency.
   - Benchmark with live LLM calls to capture end-to-end latency.
2. Target latency baselines:

   | Endpoint | p50 Target | p95 Target | p99 Target |
   |----------|-----------|-----------|-----------|
   | `POST /resume/analyze` | < 3s | < 5s | < 8s |
   | `POST /interview/start` | < 2s | < 4s | < 6s |
   | `POST /interview/answer` | < 3s | < 5s | < 8s |
   | `POST /ask` | < 2s | < 4s | < 6s |
   | `POST /jobs/search` | < 4s | < 7s | < 10s |
   | `POST /chat` (graph) | < 5s | < 8s | < 12s |
   | `GET /health` | < 50ms | < 100ms | < 200ms |
   | CRUD endpoints | < 200ms | < 500ms | < 1s |

3. Record results in `evaluation/baselines/performance_v1.json`.

**Tools & Libraries:**
- `pytest-benchmark` — microbenchmarking within pytest
- `httpx` — async HTTP client for endpoint benchmarking
- `time` / `asyncio` — timing utilities

**Deliverable:** `evaluation/benchmarks.py`, `evaluation/baselines/performance_v1.json`

---

### Milestone 5: Final Documentation (Week 5–6)

#### Step 5.1 — API Reference

**Objective:** Complete, accurate documentation for every endpoint.

**Actions:**
1. Create `docs/API_REFERENCE.md` covering all endpoints in `app/main.py`:
   - `POST /resume/analyze` — request/response models, example payloads
   - `POST /resume/improve` — request/response models, example payloads
   - `POST /interview/start` — session creation, question generation
   - `POST /interview/answer` — answer evaluation, scoring dimensions
   - `POST /interview/summary/{session_id}` — session summary retrieval
   - `POST /ask` — knowledge RAG query
   - `POST /jobs/search` — location-based job search
   - `POST /jobs/match` — profile-matched job search
   - `POST /applications` — CRUD operations (POST, GET, PUT, DELETE)
   - `GET /applications/{user_id}` — list applications
   - `PUT /applications/{app_id}/status` — status transition with validation
   - `POST /applications/{app_id}/follow-up` — follow-up scheduling
   - `POST /chat` — unified graph-based conversation
   - `GET /tasks/{task_id}/status` — async task polling
   - `GET /user/{user_id}/profile` — user profile retrieval
   - `GET /user/{user_id}/context` — user context loading
   - `GET /user/{user_id}/insights` — conversation insights
   - `GET /health` — system health check
2. For each endpoint document:
   - HTTP method and path
   - Request model (Pydantic schema with field descriptions)
   - Response model (Pydantic schema with field descriptions)
   - Example request (curl and Python `httpx`)
   - Example response (JSON)
   - Error responses (4xx, 5xx with error schemas)
   - Authentication requirements
   - Rate limiting behavior
3. Ensure OpenAPI spec is auto-generated and accessible at `/docs` and `/redoc`.

**Deliverable:** `docs/API_REFERENCE.md`

#### Step 5.2 — Architecture Guide

**Objective:** Visual and narrative documentation of the system architecture.

**Actions:**
1. Create `docs/ARCHITECTURE.md` with:
   - **System Overview Diagram** — showing all 5 Docker services and their connections (MySQL, Redis, ChromaDB, App, Celery Worker).
   - **Agent Architecture Diagram** — showing the 6 agents, their inputs/outputs, and LLM interactions.
   - **LangGraph Workflow Diagram** — showing the 8-node StateGraph (memory_load → router → conditional → specialist agents → summary → memory_save → END).
   - **RAG Pipeline Diagram** — showing document ingestion (markdown → 800-token chunks → text-embedding-3-small → ChromaDB) and query flow.
   - **Middleware Stack Diagram** — showing request flow through RequestID → Validation → Auth → RateLimit → CORS → Handler.
   - **Async Task Flow Diagram** — showing Celery task lifecycle (submit → Redis broker → worker → progress → result).
2. Use Mermaid syntax for all diagrams (renderable in GitHub Markdown).
3. Include narrative descriptions for each component explaining design decisions.

**Deliverable:** `docs/ARCHITECTURE.md`

#### Step 5.3 — Deployment Runbook

**Objective:** Step-by-step operational guide for deployment and maintenance.

**Actions:**
1. Create `docs/DEPLOYMENT_RUNBOOK.md` covering:
   - **Prerequisites:** Docker, Docker Compose, `.env` configuration, API keys
   - **First-time setup:** Database initialization, ChromaDB vectorstore creation, Celery worker verification
   - **Standard deployment:** `docker-compose up -d`, health check verification, log monitoring
   - **Configuration reference:** All environment variables from `app/config.py` with descriptions, defaults, and valid ranges
   - **Monitoring:** Health endpoint usage, structured log interpretation, correlation ID tracing
   - **Troubleshooting:** Common failure modes and resolution steps:
     - MySQL connection failures → check `MYSQL_HOST`, `MYSQL_PORT`, container health
     - ChromaDB initialization failures → verify `CHROMA_HOST`, rebuild vectorstore
     - Redis/Celery issues → check `REDIS_URL`, worker status, task queue depth
     - LLM timeout/errors → check `OPENAI_API_KEY`, rate limits, model availability
     - Rate limiting issues → adjust `RATE_LIMIT_PER_MINUTE` in config
   - **Scaling considerations:** Celery worker concurrency, database connection pooling, ChromaDB persistence
   - **Backup & recovery:** MySQL dump procedures, ChromaDB persistence volume, Redis data
   - **Rollback procedure:** Container version pinning, environment variable rollback

**Deliverable:** `docs/DEPLOYMENT_RUNBOOK.md`

#### Step 5.4 — User Guide & Demo Materials

**Objective:** End-user documentation and demonstration walkthrough.

**Actions:**
1. Create `docs/USER_GUIDE.md`:
   - **Getting Started:** How to interact with the API (authentication, base URL, rate limits)
   - **Resume Analysis Workflow:** Step-by-step guide with example payloads and expected responses
   - **Mock Interview Workflow:** Starting a session, answering questions, getting feedback, session summary
   - **Career Q&A:** How to ask questions, understanding source citations and confidence scores
   - **Job Search:** Location-based search, profile matching, understanding match scores
   - **Application Tracking:** Creating, updating, status transitions, follow-up reminders
   - **Unified Chat:** Using the `/chat` endpoint for natural conversation routing
2. Create `docs/DEMO_SCRIPT.md`:
   - **5-minute demo script** with exact API calls and expected outputs
   - Covers the "happy path" for each major feature
   - Includes talking points for each feature demonstration
   - Notes on what to highlight (multi-agent routing, RAG grounding, memory persistence)

**Deliverable:** `docs/USER_GUIDE.md`, `docs/DEMO_SCRIPT.md`

#### Step 5.5 — Update Existing Documentation

**Objective:** Ensure all existing docs reflect the final state of the project.

**Actions:**
1. Review and update `docs/PLAN.md` — mark Phase 5 as complete.
2. Review and update `docs/IMPLEMENTATION_CHECKLIST.md` — check all Phase 5 items.
3. Review and update `README.md`:
   - Ensure setup instructions are accurate.
   - Add links to all new documentation.
   - Update the feature list to reflect evaluation capabilities.
   - Add a "Project Status" badge or section.
4. Verify `docs/IMPLEMENTATION_PLAN.md` and `docs/PHASE4_DEPLOYMENT_ROADMAP.md` are consistent with implemented reality.

**Deliverable:** Updated existing documentation files.

---

## Evaluation & Metrics

### Key Performance Indicators (KPIs)

| Category | KPI | Target | Measurement Method |
|----------|-----|--------|-------------------|
| **Agent Quality** | Resume analysis accuracy | ≥ 80% alignment with human annotation | Evaluation framework scoring |
| **Agent Quality** | Interview question relevance | ≥ 85% rated as role-appropriate | Human evaluation on dataset |
| **Agent Quality** | Knowledge answer faithfulness | ≥ 90% grounded in RAG sources | Source attribution verification |
| **Agent Quality** | Router classification accuracy | ≥ 90% correct intent classification | Ground-truth dataset comparison |
| **Agent Quality** | Job search result relevance | ≥ 75% of listings match criteria | Criteria match scoring |
| **Optimization** | Prompt improvement | ≥ 15% improvement over baseline | Evaluation pipeline delta |
| **Performance** | API p95 latency (single agent) | < 5 seconds | Benchmark suite |
| **Performance** | API p95 latency (graph) | < 10 seconds | Benchmark suite |
| **Reliability** | Test pass rate | 100% (all tests green) | pytest execution |
| **Reliability** | Error handling coverage | Zero unhandled exceptions | Error audit + stress testing |
| **Documentation** | API endpoint coverage | 100% of endpoints documented | Manual review |
| **A/B Testing** | Experiment completion rate | ≥ 4 of 5 experiments reach significance | Statistical analysis |

### Metrics Collection Architecture

```
User Request
    │
    ▼
┌─────────────────────┐
│   FastAPI Endpoint   │──── Timing: request_start, request_end
│   (main.py)         │──── Logging: correlation_id, user_id, endpoint
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   LangGraph Workflow │──── Timing: per-node execution time
│   (workflow.py)     │──── Logging: agent_name, intent, confidence
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Agent Execution    │──── Quality: output scoring via evaluation framework
│   (agents/*.py)     │──── Timing: LLM call duration
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Evaluation Store   │──── Storage: evaluation/results/*.json
│   (evaluation/)     │──── Analysis: statistical significance, trend tracking
└─────────────────────┘
```

### Evaluation Cadence

| Activity | Frequency | Trigger |
|----------|-----------|---------|
| Full evaluation pipeline | After each prompt revision | Manual or CI |
| Prompt regression tests | Every test run | pytest execution |
| Performance benchmarks | Weekly during Phase 5 | Scheduled script |
| A/B experiment check | Daily during active experiments | Automated report |
| Error handling audit | Once (Milestone 4) | Manual review |

---

## A/B Testing Strategy

### Design Principles

1. **Deterministic assignment:** Users always see the same variant for a given experiment (hash-based).
2. **Isolation:** Only one experiment per agent at a time to avoid interaction effects.
3. **Minimum sample size:** Calculate upfront using power analysis; do not conclude early.
4. **Guardrails:** If any variant causes error rate > 5%, auto-disable and revert to control.

### Experiment Lifecycle

```
1. DESIGN
   ├── Define hypothesis (e.g., "CoT prompting improves evaluation fairness")
   ├── Select primary metric and minimum detectable effect
   ├── Calculate required sample size
   └── Configure traffic split

2. DEPLOY
   ├── Register experiment in ABTestManager
   ├── Implement variant routing in agent code
   └── Begin traffic routing

3. MONITOR
   ├── Daily metric collection
   ├── Error rate monitoring (guardrail check)
   └── Sample size progress tracking

4. ANALYZE
   ├── Run statistical tests when sample size is reached
   ├── Calculate confidence intervals
   ├── Apply multiple comparison correction (if >2 variants)
   └── Generate experiment report

5. DECIDE
   ├── Statistical significance + practical significance → ADOPT
   ├── Significant but negligible effect → KEEP CONTROL
   ├── Not significant → KEEP CONTROL or EXTEND
   └── Update prompt version and evaluation baseline
```

### Traffic Management

- **Default split:** 50/50 for two-variant experiments.
- **Canary split:** 90/10 for risky changes (e.g., model parameter changes).
- **Multi-variant split:** Equal distribution (33/33/34 for 3 variants).
- **Ramp-up:** Start at 10% traffic for new variant, increase to 50% after 24h if error rate is acceptable.

### Statistical Rigor

- **Significance level (α):** 0.05
- **Statistical power (1-β):** 0.80
- **Minimum detectable effect:** 10% relative improvement in primary metric
- **Test type:** Two-tailed (for two variants), Bonferroni-corrected (for >2 variants)
- **Stopping rules:** Do NOT peek at results before reaching minimum sample size. Pre-register analysis plan.

---

## Documentation Plan

### Deliverable Matrix

| Document | Location | Format | Audience | Priority |
|----------|----------|--------|----------|----------|
| API Reference | `docs/API_REFERENCE.md` | Markdown | Developers | P0 |
| Architecture Guide | `docs/ARCHITECTURE.md` | Markdown + Mermaid | Developers, Architects | P0 |
| Deployment Runbook | `docs/DEPLOYMENT_RUNBOOK.md` | Markdown | DevOps, SRE | P0 |
| User Guide | `docs/USER_GUIDE.md` | Markdown | End Users | P1 |
| Demo Script | `docs/DEMO_SCRIPT.md` | Markdown | Stakeholders, Sales | P1 |
| Evaluation README | `evaluation/README.md` | Markdown | Developers | P1 |
| Phase 5 Plan (this doc) | `docs/PHASE5_IMPLEMENTATION_PLAN.md` | Markdown | Project Management | P0 |
| Updated README | `README.md` | Markdown | All | P0 |
| Updated Checklist | `docs/IMPLEMENTATION_CHECKLIST.md` | Markdown | Project Management | P1 |

### Documentation Quality Standards

- All API examples must be copy-pasteable and tested.
- All Mermaid diagrams must render correctly on GitHub.
- All configuration references must match the actual `app/config.py` settings.
- No placeholder or TODO sections in final documentation.
- Peer review required for P0 documents before marking Phase 5 complete.

---

## Risks & Assumptions

### Assumptions

| # | Assumption | Impact if Wrong |
|---|-----------|-----------------|
| A1 | OpenAI GPT-4o-mini remains available and pricing is stable | Would need to re-evaluate model selection and prompt optimization |
| A2 | Existing test infrastructure (pytest, mocking) is sufficient for evaluation | May need additional testing frameworks |
| A3 | Current dataset size (evaluation corpus) is sufficient for statistical significance | May need to generate synthetic data or extend collection period |
| A4 | LLM-as-judge evaluation correlates with human evaluation | May need human annotation pass for calibration |
| A5 | Current Docker infrastructure handles evaluation workload | May need to adjust resource limits or run evaluation separately |

### Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|-----------|
| R1 | **LLM output non-determinism** makes evaluation scores inconsistent across runs | High | Medium | Run each evaluation 3× and average; set `temperature=0` for evaluation runs; use seed parameter if available |
| R2 | **Prompt optimization overfits** to evaluation dataset | Medium | High | Hold out 20% of evaluation data as a test set; never optimize against test data; use diverse dataset sources |
| R3 | **A/B test duration exceeds timeline** due to insufficient traffic | Medium | Medium | Use simulated traffic from evaluation datasets; accept wider confidence intervals with smaller samples |
| R4 | **Breaking changes** introduced during optimization destabilize existing passing tests | Medium | High | Run full test suite after every change; prompt regression tests as gate; git branching strategy per experiment |
| R5 | **OpenAI API rate limits** encountered during batch evaluation runs | Medium | Low | Implement exponential backoff; batch requests; run evaluation during off-peak hours; budget for evaluation API costs |
| R6 | **Evaluation framework scope creep** delays documentation work | Medium | Medium | Strict time-boxing: Milestones 1–3 must complete by end of Week 4; documentation has dedicated Weeks 5–6 |
| R7 | **RAG evaluation complexity** — assessing knowledge agent requires ground-truth answers that are subjective | Medium | Medium | Use factual questions from career guides where answers are verifiable; supplement with LLM-as-judge for subjective questions |

### Contingency Plans

- **If timeline slips:** Prioritize P0 documentation and test fixes (Milestones 4–5) over A/B testing (Milestone 3). A/B infrastructure can be delivered as a framework without completing all 5 experiments.
- **If LLM costs exceed budget:** Reduce evaluation dataset size; use GPT-4o-mini (current) rather than upgrading; cache LLM responses for repeated evaluations.
- **If evaluation scores are unexpectedly low:** Focus prompt optimization on the weakest agent first; consider whether evaluation criteria need recalibration before assuming agent failure.

---

## Timeline Summary

```
Week 1  ┃ M1: Evaluation datasets + scoring pipeline
Week 2  ┃ M1: Baselines │ M2: Prompt inventory + optimization begins
Week 3  ┃ M2: Prompt optimization + regression tests │ M3: A/B infrastructure
Week 4  ┃ M3: Experiments + analysis │ M4: Test fixes + error audit + benchmarks
Week 5  ┃ M5: API Reference + Architecture Guide + Deployment Runbook
Week 6  ┃ M5: User Guide + Demo Script + Doc updates │ Final review + sign-off
```

### Exit Criteria for Phase 5

- [ ] All 6 agents have evaluation scores recorded (baseline + optimized).
- [ ] ≥15% improvement in lowest-scoring evaluation dimensions.
- [ ] A/B testing infrastructure is functional with ≥1 completed experiment.
- [ ] All 83+ tests pass (0 failures).
- [ ] Error handling audit complete with no unhandled exceptions identified.
- [ ] Performance baselines recorded for all endpoints.
- [ ] All P0 documentation deliverables complete and peer-reviewed.
- [ ] README.md updated with final project status and documentation links.
- [ ] `docs/IMPLEMENTATION_CHECKLIST.md` updated with all Phase 5 items checked.
