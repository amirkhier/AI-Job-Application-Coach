# Evaluation Framework

This directory contains the Phase 5 evaluation, optimization, and testing
infrastructure for the AI Job Application Coach.

## Structure

```
evaluation/
├── __init__.py                # Package marker
├── README.md                  # This file
├── scoring.py                 # Agent output scoring engine
├── evaluator.py               # Orchestrator — runs agents against datasets
├── reporters.py               # Markdown / JSON report generation
├── prompt_manager.py          # Prompt versioning & variant loading
├── ab_testing.py              # A/B experiment manager
├── statistics.py              # Statistical significance testing
├── experiment_reporter.py     # Experiment result reports
├── benchmarks.py              # Endpoint latency benchmarking
├── run_regression.py          # CI-compatible prompt regression runner
├── datasets/                  # Curated evaluation datasets per agent
│   ├── README.md
│   ├── resume/
│   ├── interview/
│   ├── knowledge/
│   ├── job_search/
│   └── router/
├── prompts/                   # Versioned prompt templates per agent
│   ├── resume_prompts.py
│   ├── interview_prompts.py
│   ├── knowledge_prompts.py
│   ├── memory_prompts.py
│   ├── job_search_prompts.py
│   ├── router_prompts.py
│   └── summary_prompts.py
├── experiments/               # A/B experiment configs
├── baselines/                 # Baseline & optimised score snapshots
└── results/                   # Experiment output files
```

## Quick Start

```bash
# Run full evaluation against all agents
python -m evaluation.evaluator --all

# Run evaluation for a single agent
python -m evaluation.evaluator --agent resume

# Run prompt regression tests
python -m evaluation.run_regression

# Run benchmarks
python -m evaluation.benchmarks

# Generate experiment report
python -m evaluation.experiment_reporter --experiment resume_prompt_v2
```

## Scoring Dimensions

Each agent is scored across multiple quality dimensions (see `scoring.py`
for the full schema).  Scores are 0–10 floats.

| Agent       | Dimensions                                                   |
|-------------|--------------------------------------------------------------|
| Resume      | accuracy, completeness, actionability, ats_relevance         |
| Interview   | question_relevance, evaluation_fairness, feedback_quality, summary_accuracy |
| Knowledge   | factual_accuracy, source_attribution, confidence_calibration, completeness |
| Job Search  | location_accuracy, listing_relevance, match_scoring, fallback_quality |
| Router      | classification_accuracy, confidence_calibration, session_override, fallback_consistency |
| Memory      | context_recall, summary_quality, profile_extraction, merge_correctness |
