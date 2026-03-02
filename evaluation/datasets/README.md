# Evaluation Datasets

This directory contains curated evaluation datasets for each agent
in the AI Job Application Coach system.

## Schema

Every dataset file is a JSON array of test cases. Each test case follows:

```json
{
  "id": "resume_001",
  "description": "Senior backend engineer resume with job description",
  "input": { ... },
  "expected": {
    "dimensions": {
      "accuracy": { "min_score": 6.0, "notes": "Should identify Python strength" },
      "completeness": { "min_score": 7.0, "notes": "All sections covered" }
    }
  },
  "tags": ["positive", "senior", "backend"],
  "difficulty": "standard"
}
```

## Directories

| Directory     | Agent        | Test Cases | Coverage |
|---------------|-------------|-----------|----------|
| `resume/`     | ResumeAgent | 20+       | analyze_resume, suggest_improvements |
| `interview/`  | InterviewAgent | 30+    | generate_questions, evaluate_answer, session_summary |
| `knowledge/`  | KnowledgeAgent | 40+    | answer_question, get_topic_summary |
| `job_search/` | JobSearchAgent | 15+    | search_jobs, search_jobs_with_matching |
| `router/`     | RouterAgent    | 50+    | classify_intent |

## Tags

- `positive` — well-formed input that should produce high-quality output
- `negative` — adversarial or empty input testing error handling
- `edge_case` — boundary conditions (very short, very long, multilingual)
- `senior` / `junior` / `mid` — experience-level specific tests
- `with_jd` / `no_jd` — with or without job description context
