# Frontend Mockup Planning Document

## Overview

This document defines a step-by-step plan for designing frontend mockups for the **AI Job Application Coach** before implementation begins. The system is a multi-agent AI career coaching platform backed by a FastAPI REST API (25 endpoints), LangGraph orchestration (6 agents), RAG-powered knowledge retrieval, persistent user memory, and asynchronous task processing.

The mockups must faithfully represent every user-facing capability exposed by the API, provide a coherent end-to-end user experience, and serve as the single source of truth for the engineering team that will build the frontend.

### Goals

| # | Goal | Rationale |
|---|------|-----------|
| 1 | Map every API endpoint to a visible UI surface | No orphaned backend capability |
| 2 | Define a unified chat-first experience with structured views | Matches the `/chat` unified endpoint while exposing domain-specific tools |
| 3 | Design for desktop-first, mobile-responsive | Professional users will primarily use desktop; mobile is secondary for on-the-go status checks |
| 4 | Keep mockups high-fidelity enough for developer handoff | Reduce ambiguity during implementation |
| 5 | Identify all component boundaries early | Enable parallel frontend development |

### Tech Assumptions for Mockups

The mockups are technology-agnostic but assume:

- A modern SPA framework (React / Next.js / Vue)
- Component-based architecture
- CSS Grid / Flexbox layout
- LTR (left-to-right) text direction — see [RTL Considerations](#rtl-considerations) for internationalization notes
- REST communication with the existing FastAPI backend at `http://localhost:8000`
- Token-based or API-key authentication (`X-API-Key` header)

---

## UI Scope

### Endpoint-to-Screen Mapping

Every API endpoint must surface in at least one UI screen or component:

| API Endpoint | Method | UI Screen / Component | Priority |
|-------------|--------|----------------------|----------|
| `GET /health` | GET | Status indicator in top nav (green/yellow dot) | P0 |
| `POST /chat` | POST | **Chat Screen** — primary interaction surface | P0 |
| `POST /resume` | POST | **Resume Analyzer** — structured form + results | P0 |
| `POST /resume/improve` | POST | **Resume Improver** — side-by-side before/after | P0 |
| `POST /resume/audit` | POST | **Resume Audit** — async trigger + progress bar | P1 |
| `POST /interview/start` | POST | **Interview Practice** — session launcher | P0 |
| `POST /interview/answer` | POST | **Interview Practice** — answer submission + feedback | P0 |
| `GET /interview/questions/{job_title}` | GET | **Quick Questions** — question list view | P1 |
| `POST /interview/report` | POST | **Interview Report** — async trigger | P1 |
| `POST /ask` | POST | **Knowledge Q&A** — question + sourced answer | P0 |
| `POST /jobs/search` | POST | **Job Search** — search form + results grid | P0 |
| `POST /jobs/match` | POST | **Job Match** — resume-to-job fit scores | P1 |
| `GET /jobs/location/{city}` | GET | **Job Search** — map / location card | P2 |
| `POST /applications` | POST | **Application Tracker** — create form | P0 |
| `GET /applications` | GET | **Application Tracker** — list / board view | P0 |
| `PUT /applications/{id}` | PUT | **Application Tracker** — inline edit / modal | P0 |
| `DELETE /applications/{id}` | DELETE | **Application Tracker** — delete confirmation | P0 |
| `GET /applications/follow-ups` | GET | **Application Tracker** — follow-up badge/filter | P1 |
| `POST /applications/batch-update` | POST | **Application Tracker** — bulk action bar | P2 |
| `GET /user/{id}/profile` | GET | **Profile Screen** — view/edit profile | P0 |
| `POST /user/{id}/profile/update` | POST | **Profile Screen** — save profile | P0 |
| `GET /user/{id}/context` | GET | **Profile Screen** — context summary tab | P1 |
| `GET /user/{id}/insights` | GET | **Insights Dashboard** — patterns & recommendations | P1 |
| `GET /tasks/{id}/status` | GET | **Task Status** — global async task tracker | P1 |
| `GET /result/{id}` | GET | **Task Status** — result viewer | P1 |

### Screen Inventory

Based on the mapping above, the application requires **9 distinct screens** plus shared layout components:

| # | Screen | Description |
|---|--------|-------------|
| 1 | **Login / API Key Setup** | One-time API key configuration; no traditional auth flow |
| 2 | **Dashboard** | Landing page with summary cards, recent activity, and quick-action buttons |
| 3 | **Chat** | Conversational interface — the primary way users interact with all agents |
| 4 | **Resume Studio** | Resume analysis, improvement, and async audit in a structured workspace |
| 5 | **Interview Practice** | Session-based mock interview with real-time feedback |
| 6 | **Knowledge Base** | Career Q&A with source-attributed answers |
| 7 | **Job Search** | Search, filter, match, and location-based results |
| 8 | **Application Tracker** | Kanban board / list view for tracking application pipeline |
| 9 | **Profile & Insights** | User profile management, conversation history, AI-generated insights |

---

## User Flow

### Primary User Flow (Happy Path)

```
┌──────────────┐     ┌────────────┐     ┌───────────────────────────┐
│  Login /     │────▶│  Dashboard │────▶│  Choose Feature           │
│  API Key     │     │            │     │  (Chat / Resume / etc.)   │
└──────────────┘     └────────────┘     └───────────┬───────────────┘
                                                    │
                     ┌──────────────────────────────┼──────────────────────────────┐
                     │              │               │               │              │
                     ▼              ▼               ▼               ▼              ▼
              ┌────────────┐ ┌───────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────┐
              │   Chat     │ │  Resume   │ │  Interview   │ │   Job    │ │ Application  │
              │  (unified) │ │  Studio   │ │  Practice    │ │  Search  │ │   Tracker    │
              └─────┬──────┘ └─────┬─────┘ └──────┬───────┘ └─────┬────┘ └──────┬───────┘
                    │              │               │               │              │
                    │              ▼               ▼               ▼              │
                    │       ┌────────────┐  ┌───────────┐  ┌───────────┐         │
                    │       │ View       │  │ Answer     │  │ Save to   │◀────────┘
                    │       │ Results /  │  │ Questions  │  │ Tracker   │
                    │       │ Improve    │  │ + Feedback │  │           │
                    │       └────────────┘  └───────────┘  └───────────┘
                    │
                    ▼
              ┌────────────────────────────────────────┐
              │  All agent responses flow through      │
              │  the chat interface OR structured views │
              └────────────────────────────────────────┘
```

### Flow Details

#### Flow 1: First-Time Setup
1. User opens the app → **Login / API Key Setup** screen
2. User enters API key (or server URL if self-hosted) → key stored in local storage
3. App calls `GET /health` to verify connectivity → redirects to **Dashboard**

#### Flow 2: Chat-First Interaction
1. From **Dashboard** → user clicks "Start Chatting" → **Chat** screen
2. User types a natural-language message (e.g., "Review my resume")
3. App sends `POST /chat` with message + optional attachments (resume text, job description)
4. Response displayed in chat bubble with structured data (scores, feedback) rendered inline
5. User continues conversation; system maintains context via `session_id`

#### Flow 3: Resume Workflow
1. From **Dashboard** or **Sidebar** → user navigates to **Resume Studio**
2. User pastes resume text (+ optional job description) → clicks "Analyze"
3. App calls `POST /resume` → results displayed: score gauge, strengths/weaknesses cards, ATS panel, keyword analysis
4. User clicks "Improve" → `POST /resume/improve` → side-by-side before/after bullet comparison
5. User clicks "Full Audit" → `POST /resume/audit` → async task ID returned → progress bar shown
6. App polls `GET /tasks/{id}/status` → on completion, displays detailed audit report

#### Flow 4: Interview Practice
1. From **Dashboard** or **Sidebar** → user navigates to **Interview Practice**
2. User selects role, level, question count → clicks "Start Interview"
3. App calls `POST /interview/start` → first question displayed in a card
4. User types answer → clicks "Submit" → `POST /interview/answer`
5. Feedback card appears: score, strengths, improvements, suggested improvement
6. Next question appears automatically (or user clicks "Next")
7. After final question → session summary displayed with aggregate scores, weak areas, overall assessment
8. User clicks "Generate Report" → `POST /interview/report` → async report

#### Flow 5: Career Knowledge Q&A
1. From **Dashboard** or **Sidebar** → user navigates to **Knowledge Base**
2. User types career question → clicks "Ask"
3. App calls `POST /ask` → answer displayed with source badges, relevance score, and related topic chips
4. User clicks a related topic chip → new query auto-populated

#### Flow 6: Job Search
1. From **Dashboard** or **Sidebar** → user navigates to **Job Search**
2. User fills in query, location, experience level, remote preference → clicks "Search"
3. App calls `POST /jobs/search` → job cards displayed in a grid with salary, skills tags, match score
4. Location card shows geocoded city info + nearby companies
5. User clicks "Match Against My Resume" → `POST /jobs/match` → match scores overlaid on job cards
6. User clicks "Save to Tracker" on a job card → creates application via `POST /applications`

#### Flow 7: Application Tracking
1. From **Dashboard** or **Sidebar** → user navigates to **Application Tracker**
2. Kanban board shows columns: Applied → Interviewing → Offer → Rejected → Withdrawn
3. User drags a card between columns → `PUT /applications/{id}` with new status (validated against status workflow)
4. User clicks a card → detail modal with notes, follow-up date, edit form
5. Follow-up badges highlight overdue follow-ups (`GET /applications/follow-ups`)
6. Bulk selection checkbox → "Batch Update" button → `POST /applications/batch-update`

#### Flow 8: Profile & Insights
1. From **Sidebar** → user navigates to **Profile & Insights**
2. Profile tab: displays user skills, experience level, preferences (from `GET /user/{id}/profile`)
3. User edits fields → clicks "Save" → `POST /user/{id}/profile/update`
4. Context tab: conversation history summary (from `GET /user/{id}/context`)
5. Insights tab: AI-generated patterns, agent usage chart, recommendations (from `GET /user/{id}/insights`)

---

## Screen-by-Screen Mockup Plan

### Screen 1: Login / API Key Setup

**Purpose:** Authenticate and verify backend connectivity.

**Layout:**
- Centered card on a branded background
- Logo + tagline at top
- Input field for API key (masked, with show/hide toggle)
- Optional: server URL input (default: `http://localhost:8000`)
- "Connect" button
- Status indicator: connecting → connected (green) / failed (red) with error message
- "Continue without key" link (if `API_KEY` is not configured server-side)

**Behavior:**
- On submit, call `GET /health` with `X-API-Key` header
- If `401`, show "Invalid API key"
- If `200`, store key in `localStorage` and redirect to Dashboard
- If connection error, show "Cannot reach server at {url}"

**Responsive:** Same layout on mobile, full-width card

---

### Screen 2: Dashboard

**Purpose:** Overview of current state, quick actions, and recent activity.

**Layout (Desktop — 12-column grid):**

```
┌──────────────────────────────────────────────────────────┐
│  Top Navigation Bar                                      │
│  [Logo] [Dashboard] [Chat] [Resume] [Interview]         │
│         [Jobs] [Tracker] [Profile]   [Health: ●]        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │ Resume      │ │ Interview   │ │ Applications │       │
│  │ Score: 7.5  │ │ Sessions: 3 │ │ Active: 8    │       │
│  │ [Analyze →] │ │ [Practice→] │ │ [View →]     │       │
│  └─────────────┘ └─────────────┘ └─────────────┘       │
│                                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │ Job Search  │ │ Knowledge   │ │ Quick Chat   │       │
│  │ [Search →]  │ │ [Ask →]     │ │ [Chat →]     │       │
│  └─────────────┘ └─────────────┘ └─────────────┘       │
│                                                          │
│  ┌──────────────────────────────────────────────┐       │
│  │  Recent Activity Feed                         │       │
│  │  • Resume analyzed — Score: 7.5 (2h ago)      │       │
│  │  • Interview session — Backend Eng (yesterday) │       │
│  │  • Application: Google → Interviewing          │       │
│  └──────────────────────────────────────────────┘       │
│                                                          │
│  ┌──────────────────────┐ ┌─────────────────────┐       │
│  │ Upcoming Follow-ups  │ │ AI Insights          │       │
│  │ • Google — Mar 10    │ │ "Focus on system     │       │
│  │ • Meta — Mar 12      │ │  design prep"        │       │
│  └──────────────────────┘ └─────────────────────┘       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Components:**
1. **Summary Cards** (3 × top row) — Resume score, interview session count, active applications count
2. **Quick Action Cards** (3 × second row) — Direct links to Job Search, Knowledge, Chat
3. **Recent Activity Feed** — Last 5–10 interactions (from `GET /user/{id}/context`)
4. **Upcoming Follow-ups Widget** — From `GET /applications/follow-ups`
5. **AI Insights Card** — Top recommendation from `GET /user/{id}/insights`

**Data Sources:** `GET /health`, `GET /user/{id}/context`, `GET /user/{id}/insights`, `GET /applications/follow-ups`

**Responsive (Mobile):** Stack cards vertically, single column. Collapse to 3 summary cards → activity feed → follow-ups.

---

### Screen 3: Chat

**Purpose:** Unified conversational interface that routes to all agents via `POST /chat`.

**Layout (Desktop):**

```
┌──────────────────────────────────────────────────────────┐
│  [← Back]  AI Career Coach Chat           [New Chat]     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │               Message History                     │   │
│  │                                                   │   │
│  │  [User]: Can you review my resume?                │   │
│  │                                                   │   │
│  │  [Coach]: I've analyzed your resume. Here's       │   │
│  │  what I found:                                    │   │
│  │  ┌────────────────────────────────────────┐       │   │
│  │  │ Score: 7.5/10  ██████████░░░           │       │   │
│  │  │ Strengths: Quantified achievements     │       │   │
│  │  │ Weaknesses: Missing summary section    │       │   │
│  │  │ [View Full Analysis →]                 │       │   │
│  │  └────────────────────────────────────────┘       │   │
│  │                                                   │   │
│  │  [User]: Start an interview for senior BE         │   │
│  │                                                   │   │
│  │  [Coach]: Let's begin your mock interview.        │   │
│  │  ┌────────────────────────────────────────┐       │   │
│  │  │ Q1 (Behavioral / Hard):                │       │   │
│  │  │ "Tell me about a time you led a        │       │   │
│  │  │  complex migration project."           │       │   │
│  │  │ Key Points: Leadership, Complexity     │       │   │
│  │  └────────────────────────────────────────┘       │   │
│  │                                                   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Attachment Bar                                   │   │
│  │  [📄 Resume] [📋 Job Desc] [🎯 Role/Level]       │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  [Type your message...]                    [Send] │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  Sidebar (collapsible):                                  │
│  ┌──────────────┐                                       │
│  │ Active       │                                       │
│  │ Session Info │                                       │
│  │              │                                       │
│  │ Intent:      │                                       │
│  │ resume_anal  │                                       │
│  │              │                                       │
│  │ Agents Used: │                                       │
│  │ router,      │                                       │
│  │ resume,      │                                       │
│  │ memory       │                                       │
│  │              │                                       │
│  │ Confidence:  │                                       │
│  │ 95%          │                                       │
│  │              │                                       │
│  │ Time: 5.2s   │                                       │
│  └──────────────┘                                       │
└──────────────────────────────────────────────────────────┘
```

**Components:**
1. **Message List** — Scrollable thread with user and AI messages
2. **Rich Response Cards** — Inline structured data renders (score gauges, tables, charts) embedded in AI messages
3. **Attachment Bar** — Toggle panels to attach resume text, job description, interview role/level
4. **Input Bar** — Text input + send button (Ctrl+Enter to send)
5. **Session Sidebar** (desktop only, collapsible) — Shows current intent, agents used, confidence, processing time, interview session ID
6. **New Chat Button** — Resets `session_id`

**Rich Card Types to Design:**
- Resume Analysis Card (score gauge, strength/weakness pills, ATS compatibility bar)
- Resume Improvement Card (original vs improved bullets in two columns)
- Interview Question Card (type badge, difficulty badge, key points list)
- Interview Feedback Card (score, strengths, improvements)
- Interview Session Summary Card (aggregate table)
- Knowledge Answer Card (answer, source badges, relevance bar, related topic chips)
- Job Listing Card (title, company, salary, skill tags, remote badge)
- Application Created Confirmation Card
- Async Task Submitted Card (with "View Progress" link)

**API:** `POST /chat` (primary), with optional fields: `resume_text`, `job_description`, `interview_role`, `interview_level`, `job_search_location`, `interview_session_id`, `interview_answer`, `interview_question_id`

**Responsive (Mobile):** Full-screen chat, sidebar hidden (accessible via swipe or hamburger), attachment bar collapses to a "+" button that opens a bottom sheet.

---

### Screen 4: Resume Studio

**Purpose:** Structured workspace for resume analysis, improvement, and auditing.

**Layout (Desktop — 3-panel):**

```
┌────────────────────────────────────────────────────────────┐
│  Resume Studio                          [Audit (async) ▶]  │
├──────────────┬───────────────────────┬─────────────────────┤
│              │                       │                     │
│  Input Panel │   Results Panel       │   Details Panel     │
│  (left 30%)  │   (center 40%)       │   (right 30%)      │
│              │                       │                     │
│  ┌────────┐  │  ┌─────────────────┐  │  ┌───────────────┐ │
│  │Resume  │  │  │ Overall Score   │  │  │ ATS Compat.   │ │
│  │Text    │  │  │    7.5 / 10     │  │  │ Score: 8.0    │ │
│  │        │  │  │  ████████░░     │  │  │ Issues: 0     │ │
│  │(text-  │  │  └─────────────────┘  │  │ Suggestions:  │ │
│  │area)   │  │                       │  │ • Add keywords│ │
│  │        │  │  ┌─────────────────┐  │  └───────────────┘ │
│  │        │  │  │ Strengths       │  │                     │
│  │        │  │  │ ✓ Quantified    │  │  ┌───────────────┐ │
│  └────────┘  │  │ ✓ Technical     │  │  │ Keywords      │ │
│              │  └─────────────────┘  │  │ ✓ Python      │ │
│  ┌────────┐  │                       │  │ ✓ AWS         │ │
│  │Job Desc│  │  ┌─────────────────┐  │  │ ✗ CI/CD      │ │
│  │(text-  │  │  │ Weaknesses      │  │  │ ✗ Terraform  │ │
│  │area)   │  │  │ ✗ No summary    │  │  └───────────────┘ │
│  │        │  │  │ ✗ Missing certs │  │                     │
│  └────────┘  │  └─────────────────┘  │  ┌───────────────┐ │
│              │                       │  │ Section       │ │
│  [Analyze]   │  ┌─────────────────┐  │  │ Feedback      │ │
│  [Improve]   │  │ Recommendations │  │  │ Contact: ✓    │ │
│              │  │ 1. Add summary  │  │  │ Summary: ✗    │ │
│              │  │ 2. Add metrics  │  │  │ Experience: ✓ │ │
│              │  └─────────────────┘  │  └───────────────┘ │
│              │                       │                     │
├──────────────┴───────────────────────┴─────────────────────┤
│  Improvement View (toggle)                                  │
│  ┌──────────────────────┬────────────────────────────────┐ │
│  │ ORIGINAL              │ IMPROVED                       │ │
│  │ Led migration of      │ Led cloud migration of         │ │
│  │ monolithic app         │ monolith to 12 microservices,  │ │
│  │                        │ reducing deploy time by 60%    │ │
│  │                        │ Reasoning: Added metrics       │ │
│  └──────────────────────┴────────────────────────────────┘ │
│  Priority Actions: [1. Rewrite summary] [2. Quantify all]  │
└────────────────────────────────────────────────────────────┘
```

**Components:**
1. **Resume Input** — Textarea with character count (min 50 chars enforced), placeholder hint
2. **Job Description Input** — Optional textarea
3. **Action Buttons** — "Analyze" (`POST /resume`), "Improve" (`POST /resume/improve`), "Full Audit" (`POST /resume/audit`)
4. **Score Gauge** — Circular or bar chart showing overall score (0–10)
5. **Strengths/Weaknesses Lists** — Green/red pill lists
6. **Recommendations List** — Numbered actionable items
7. **ATS Compatibility Panel** — Score bar + issues + suggestions
8. **Keyword Analysis Panel** — Present (green check) / Missing (red cross) lists
9. **Section Feedback Panel** — Per-section status (contact, summary, experience, skills, education)
10. **Improvement Diff View** — Two-column original vs improved bullets with reasoning
11. **Priority Actions Bar** — Ordered chips of highest-impact changes
12. **Async Audit Progress** — Progress bar + status text when audit is running

**API Calls:** `POST /resume`, `POST /resume/improve`, `POST /resume/audit`, `GET /tasks/{id}/status`, `GET /result/{id}`

**Responsive (Mobile):** Single column. Input panel on top, results below (accordion sections). Improvement diff stacks vertically (original above improved).

---

### Screen 5: Interview Practice

**Purpose:** Multi-turn mock interview with real-time STAR-method feedback.

**Layout (Desktop):**

```
┌──────────────────────────────────────────────────────────┐
│  Interview Practice                                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Setup Panel (before session starts):                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Role: [Backend Engineer        ▼]                │   │
│  │  Level: [● Junior ● Mid ● Senior ● Lead]         │   │
│  │  Questions: [5] (slider 1–10)                     │   │
│  │                              [Start Interview ▶]  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  Session View (after session starts):                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Progress: Q2 of 5  ████░░░░░░                    │   │
│  │  Role: Senior Backend Engineer                    │   │
│  │  Session: abc-123...                              │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────┬───────────────────────────┐   │
│  │  Question Card       │  Feedback Card            │   │
│  │                      │  (appears after submit)   │   │
│  │  Q2 / Behavioral     │                           │   │
│  │  Difficulty: Hard    │  Score: 7.5/10            │   │
│  │                      │  ██████████░░░            │   │
│  │  "Describe a time    │                           │   │
│  │   you had to deal    │  Strengths:               │   │
│  │   with a critical    │  ✓ Clear STAR structure   │   │
│  │   production         │  ✓ Specific metrics       │   │
│  │   incident."         │                           │   │
│  │                      │  Improvements:            │   │
│  │  Key Points:         │  • Elaborate on team      │   │
│  │  • Incident mgmt     │    dynamics               │   │
│  │  • Communication     │                           │   │
│  │  • Resolution        │  Suggestion:              │   │
│  │                      │  "Consider adding more    │   │
│  │                      │   context about..."       │   │
│  └──────────────────────┴───────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Your Answer:                                     │   │
│  │  [                                               ]│   │
│  │  [                                               ]│   │
│  │  [                                               ]│   │
│  │                            [Submit Answer ▶]      │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  Session Summary (after final question):                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Overall Score: 7.2/10                            │   │
│  │  Questions Answered: 5/5                          │   │
│  │  Strong Areas: Leadership, Technical depth        │   │
│  │  Weak Areas: Stakeholder management               │   │
│  │  Recommendation: "Focus on communication..."      │   │
│  │                 [Generate Full Report ▶]           │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Components:**
1. **Setup Form** — Role input (text or dropdown), level radio buttons, question count slider
2. **Progress Bar** — Current question / total questions
3. **Question Card** — Question text, type badge (behavioral/technical/situational), difficulty badge, key points list
4. **Answer Textarea** — Min 10 chars, character counter
5. **Submit Button** — Disabled while awaiting response, shows spinner
6. **Feedback Card** — Score gauge, strength/improvement lists, specific feedback, suggested improvement
7. **Session Summary Card** — Aggregate scores, strong/weak areas, overall recommendation
8. **Generate Report Button** — Triggers `POST /interview/report` → async task

**State Machine:**
```
setup → question_displayed → answer_submitted → feedback_displayed → (next question | session_complete)
```

**API Calls:** `POST /interview/start`, `POST /interview/answer`, `POST /interview/report`, `GET /tasks/{id}/status`

**Responsive (Mobile):** Single column, question and feedback stack vertically. Setup form is full-width.

---

### Screen 6: Knowledge Base

**Purpose:** Career Q&A grounded in curated knowledge (RAG).

**Layout (Desktop):**

```
┌──────────────────────────────────────────────────────────┐
│  Career Knowledge Base                                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  [How should I negotiate my salary?        ] [Ask]│   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  Quick Topics:                                           │
│  [Salary Negotiation] [Interview Tips] [Resume Writing]  │
│  [Industry Insights] [Career Growth]                     │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Answer:                                          │   │
│  │  When negotiating salary, start by researching    │   │
│  │  market rates for your role and location...       │   │
│  │                                                   │   │
│  │  Sources: [salary_negotiation.md]                 │   │
│  │           [industry_insights.md]                  │   │
│  │                                                   │   │
│  │  Relevance: 87%  █████████░                       │   │
│  │                                                   │   │
│  │  Related Topics:                                  │   │
│  │  [Benefits Negotiation] [Equity Packages]         │   │
│  │  [Counter-Offers]                                 │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Previous Questions:                              │   │
│  │  • How to prepare for behavioral interviews?      │   │
│  │  • What keywords should I include in my resume?   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Components:**
1. **Search Bar** — Text input with "Ask" button, min 5 chars
2. **Quick Topic Chips** — Pre-populated career topic shortcuts
3. **Answer Card** — Rendered markdown answer text
4. **Source Badges** — Clickable badges showing which knowledge documents were used
5. **Relevance Score Bar** — Visual indicator of answer confidence
6. **Related Topic Chips** — Clickable chips that auto-populate the search bar
7. **Question History** — Recent questions asked in this session

**API:** `POST /ask`

**Responsive (Mobile):** Single column, search bar sticky at top, answer card fills screen.

---

### Screen 7: Job Search

**Purpose:** Search for jobs, view results, match against resume, and save to tracker.

**Layout (Desktop):**

```
┌────────────────────────────────────────────────────────────┐
│  Job Search                                                 │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Query: [Python backend developer           ]         │ │
│  │  Location: [Tel Aviv        ]                         │ │
│  │  Level: [● Jr ● Mid ● Sr ● Lead]  ☑ Remote OK        │ │
│  │  Results: [5 ▼]                     [Search ▶]        │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌──────────────────────┐  ┌────────────────────────────┐ │
│  │  Location Info        │  │  Nearby Companies          │ │
│  │  📍 Tel Aviv, Israel  │  │  • Waze (1.2 km)           │ │
│  │  Lat: 32.08           │  │  • CheckPoint (2.1 km)     │ │
│  │  Lon: 34.78           │  │  • Monday.com (0.8 km)     │ │
│  └──────────────────────┘  └────────────────────────────┘ │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Results (5 jobs)                    [Match Resume ▶] │ │
│  │                                                       │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │ │
│  │  │Sr Python Dev│ │Backend Lead │ │API Engineer │     │ │
│  │  │TechCorp     │ │FinStart     │ │CloudBase    │     │ │
│  │  │Tel Aviv     │ │Tel Aviv     │ │Remote       │     │ │
│  │  │$120-160k    │ │$140-180k    │ │$100-140k    │     │ │
│  │  │[🏷 Python]  │ │[🏷 Go]      │ │[🏷 FastAPI] │     │ │
│  │  │[🏷 AWS]     │ │[🏷 K8s]     │ │[🏷 AWS]     │     │ │
│  │  │[🌐 Remote]  │ │             │ │[🌐 Remote]  │     │ │
│  │  │Match: 92%   │ │Match: 78%   │ │Match: 85%   │     │ │
│  │  │[Save to     │ │[Save to     │ │[Save to     │     │ │
│  │  │ Tracker]    │ │ Tracker]    │ │ Tracker]    │     │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘     │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

**Components:**
1. **Search Form** — Query input, location input, level radio group, remote checkbox, result count dropdown
2. **Location Info Card** — City, coordinates, display name, country, found status
3. **Nearby Companies Card** — List with distance and type
4. **Job Cards Grid** — Cards with title, company, location, salary, skill tags, remote badge, match score (if available)
5. **"Match Resume" Button** — Triggers `POST /jobs/match` and overlays match scores on cards
6. **"Save to Tracker" Button** — Per-card button to create an application record

**API Calls:** `POST /jobs/search`, `POST /jobs/match`, `GET /jobs/location/{city}`, `POST /applications`

**Responsive (Mobile):** Search form stacks vertically. Job cards in single-column list. Location info and nearby companies in horizontal scroll.

---

### Screen 8: Application Tracker

**Purpose:** Kanban-style pipeline for managing job applications.

**Layout (Desktop — Kanban Board):**

```
┌──────────────────────────────────────────────────────────────────┐
│  Application Tracker              [+ New Application] [Batch ▶]  │
├──────────────────────────────────────────────────────────────────┤
│  Filter: [All ▼]  [Follow-ups Only ☐]   Search: [________]      │
├──────────────┬──────────────┬─────────────┬──────────┬──────────┤
│   APPLIED    │ INTERVIEWING │    OFFER    │ REJECTED │WITHDRAWN │
│              │              │             │          │          │
│ ┌──────────┐│ ┌──────────┐ │             │          │          │
│ │ Google   ││ │ Meta     │ │             │          │          │
│ │ Sr SWE   ││ │ Backend  │ │             │          │          │
│ │ Mar 1    ││ │ Mar 3    │ │             │          │          │
│ │ 📌 F/U:  ││ │ Notes:   │ │             │          │          │
│ │ Mar 10   ││ │ Phone    │ │             │          │          │
│ │          ││ │ screen   │ │             │          │          │
│ │ [Edit]   ││ │ done     │ │             │          │          │
│ │ [Delete] ││ │          │ │             │          │          │
│ └──────────┘│ │ [Edit]   │ │             │          │          │
│              │ │ [Delete] │ │             │          │          │
│ ┌──────────┐│ └──────────┘ │             │          │          │
│ │ Amazon   ││              │             │          │          │
│ │ SDE II   ││              │             │          │          │
│ │ Mar 4    ││              │             │          │          │
│ └──────────┘│              │             │          │          │
│              │              │             │          │          │
├──────────────┴──────────────┴─────────────┴──────────┴──────────┤
│  Status Workflow: applied → interviewing → offer                 │
│                       ↓          ↓           ↓                   │
│                   rejected    rejected    rejected                │
│                       ↓          ↓           ↓                   │
│                   withdrawn   withdrawn   withdrawn              │
└──────────────────────────────────────────────────────────────────┘
```

**Components:**
1. **Kanban Columns** — 5 columns: Applied, Interviewing, Offer, Rejected, Withdrawn
2. **Application Cards** — Company, position, date, follow-up date (with overdue badge), notes preview
3. **Drag-and-Drop** — Move cards between columns (validates against `APPLICATION_STATUS_FLOW`)
4. **Invalid Transition Toast** — Warning when drag target violates status workflow
5. **"+ New Application" Button** — Opens create modal
6. **Create/Edit Modal** — Form with company, position, job URL, date, notes, follow-up date, status dropdown
7. **Delete Confirmation Dialog** — "Are you sure?" with application details
8. **Filter Bar** — Status filter dropdown, follow-ups-only toggle, text search
9. **Batch Action Bar** — Checkbox selection + bulk status update button → `POST /applications/batch-update`
10. **Follow-up Badge** — Orange/red indicator on cards with upcoming/overdue follow-ups

**API Calls:** `GET /applications`, `POST /applications`, `PUT /applications/{id}`, `DELETE /applications/{id}`, `GET /applications/follow-ups`, `POST /applications/batch-update`

**Responsive (Mobile):** Switch from Kanban board to list view with status pills. Swipe actions for quick status change. Create/edit as full-screen modal.

---

### Screen 9: Profile & Insights

**Purpose:** View/edit user profile, browse conversation history, and review AI-generated insights.

**Layout (Desktop — 3-tab layout):**

```
┌──────────────────────────────────────────────────────────┐
│  My Profile                                               │
│  [Profile] [History] [Insights]                           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Profile Tab:                                            │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Skills: [Python] [AWS] [Docker] [+ Add]          │   │
│  │  Experience Level: ● Junior ● Mid ● Senior        │   │
│  │  Target Roles: [Backend Engineer] [+ Add]         │   │
│  │  Preferences: [Remote OK ☑] [Location: Tel Aviv]  │   │
│  │                                          [Save]   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  History Tab:                                            │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Total Conversations: 15                          │   │
│  │  Top Intents: resume_analysis (8), interview (5)  │   │
│  │                                                   │   │
│  │  Recent:                                          │   │
│  │  • [Mar 5] resume_analysis — "Reviewed resume..." │   │
│  │  • [Mar 4] interview_practice — "Mock interview"  │   │
│  │  • [Mar 3] career_advice — "Salary negotiation"   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  Insights Tab:                                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │  AI Insights:                                     │   │
│  │  • "You frequently ask about resume formatting"   │   │
│  │  • "Interview practice sessions are increasing"   │   │
│  │                                                   │   │
│  │  Recommendations:                                 │   │
│  │  • "Focus on system design preparation"           │   │
│  │  • "Consider adding cloud certifications"         │   │
│  │                                                   │   │
│  │  Agent Usage:                                     │   │
│  │  ┌──────────────────────────────────┐             │   │
│  │  │ Resume ████████████ 8            │             │   │
│  │  │ Interview ██████████ 5           │             │   │
│  │  │ Knowledge ████ 2                 │             │   │
│  │  └──────────────────────────────────┘             │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Components:**
1. **Tab Navigation** — Profile, History, Insights
2. **Profile Form** — Tags input for skills, radio for experience level, tags for target roles, checkboxes for preferences
3. **Save Button** — Triggers `POST /user/{id}/profile/update`
4. **Conversation History List** — Scrollable list with date, intent, summary preview
5. **Context Summary Card** — Total conversations, top intents
6. **Insights List** — AI-generated insight bullets
7. **Recommendations List** — Actionable suggestion bullets
8. **Agent Usage Bar Chart** — Horizontal bars showing per-agent usage counts

**API Calls:** `GET /user/{id}/profile`, `POST /user/{id}/profile/update`, `GET /user/{id}/context`, `GET /user/{id}/insights`

**Responsive (Mobile):** Tabs become a segmented control. Forms go full-width. Chart simplified.

---

## Component Breakdown

### Shared / Global Components

| Component | Description | Used In |
|-----------|------------|---------|
| `AppShell` | Top nav + sidebar + main content area | All screens |
| `TopNav` | Logo, navigation links, health indicator dot, user avatar | All screens |
| `Sidebar` | Collapsible navigation with screen icons + labels | All screens (desktop) |
| `BottomNav` | Mobile-only bottom tab bar | All screens (mobile) |
| `HealthIndicator` | Green/yellow/red dot from `GET /health` | TopNav |
| `LoadingSpinner` | Inline and overlay spinner | All screens |
| `ErrorBoundary` | Global error catch with retry button | All screens |
| `Toast` | Success/error/warning notification | All screens |
| `ConfirmDialog` | "Are you sure?" modal | Tracker delete, destructive actions |
| `EmptyState` | Illustration + message when no data | Lists, trackers |
| `AsyncTaskBanner` | Shows pending async task with polling progress | Resume audit, interview report |

### Domain Components

| Component | Props (key data) | Screen |
|-----------|-----------------|--------|
| `ScoreGauge` | `score: number, max: number, label: string` | Resume Studio, Interview, Chat |
| `StrengthWeaknessList` | `items: string[], type: 'strength' \| 'weakness'` | Resume Studio, Chat |
| `ATSPanel` | `score, issues[], suggestions[]` | Resume Studio |
| `KeywordGrid` | `present[], missing[]` | Resume Studio |
| `SectionFeedbackList` | `{section: string, feedback: string}[]` | Resume Studio |
| `BulletDiff` | `original, improved, reasoning` | Resume Studio, Chat |
| `InterviewQuestionCard` | `id, question, type, difficulty, keyPoints[]` | Interview, Chat |
| `InterviewFeedbackCard` | `score, strengths[], improvements[], feedback, suggestion` | Interview, Chat |
| `SessionSummaryCard` | `overallScore, questionsAnswered, strongAreas[], weakAreas[]` | Interview |
| `KnowledgeAnswerCard` | `answer, sources[], relevanceScore, relatedTopics[]` | Knowledge, Chat |
| `JobCard` | `title, company, location, salary, skills[], remote, matchScore` | Job Search, Chat |
| `LocationInfoCard` | `city, lat, lon, displayName, country` | Job Search |
| `NearbyCompanyList` | `{name, type, distance}[]` | Job Search |
| `ApplicationCard` | `company, position, status, date, followUp, notes` | Tracker |
| `KanbanColumn` | `status, applications[]` | Tracker |
| `ApplicationFormModal` | Full create/edit form | Tracker |
| `ChatBubble` | `message, sender, timestamp, richCard?` | Chat |
| `AttachmentBar` | `resumeText?, jobDescription?, role?, level?` | Chat |
| `AgentBadge` | `agentName, icon` | Chat sidebar |
| `TagInput` | `tags[], onAdd, onRemove` | Profile |
| `AgentUsageChart` | `{agent: string, count: number}[]` | Profile Insights |
| `ActivityFeedItem` | `intent, summary, date, agentUsed` | Dashboard, Profile |
| `FollowUpBadge` | `date, isOverdue` | Tracker, Dashboard |

---

## UX & Interaction Guidelines

### General Principles

1. **Chat-first, structured-second**: The Chat screen is the default entry point. Structured screens (Resume Studio, Interview Practice, etc.) are accessible for users who prefer form-based workflows. Both paths invoke the same backend endpoints.

2. **Progressive disclosure**: Show summary information first (scores, top-level feedback). Full details expand on click or in a detail panel.

3. **Optimistic UI for mutations**: When updating application status (drag-and-drop) or saving profile changes, update the UI immediately and reconcile on API response. Show toast on failure with an undo option.

4. **Loading states for every API call**: No blank screens. Show skeleton loaders for initial data fetches and inline spinners for mutations.

5. **Empty states with call-to-action**: If no applications exist, show an illustration with "Track your first application →". If no interviews, "Start your first mock interview →".

### Interaction Specifications

| Interaction | Behavior | Feedback |
|------------|----------|----------|
| Submit resume for analysis | Disable button, show spinner, ~3–5s | Score gauge animates in; sections fade in sequentially |
| Submit interview answer | Disable textarea + button, show spinner, ~3–5s | Feedback card slides in from right; score pulses once |
| Drag application card | Card lifts and follows cursor; valid columns highlight green; invalid columns show red border | Toast confirms new status or warns of invalid transition |
| Ask knowledge question | Show typing indicator in answer area, ~2–4s | Answer fades in; source badges pop in; related topics animate as chips |
| Search for jobs | Show skeleton job cards during loading, ~3–5s | Cards fade in; match scores overlay if matched |
| Save application from job search | Button changes to "Saved ✓" | Toast "Application saved to tracker" with link |
| Start interview session | Setup form slides out; first question slides in | Progress bar initializes |
| Full resume audit (async) | Show "Audit queued" banner with progress bar; poll every 5s | Banner updates from "Queued" → "Processing" → "Complete" with result link |
| Health check failure | Top nav indicator turns yellow/red | Tooltip explains which dependencies are down |

### Navigation Pattern

- **Desktop**: Persistent left sidebar (collapsible to icons) + top bar with breadcrumbs and health indicator
- **Mobile**: Bottom tab bar (5 tabs: Dashboard, Chat, Resume, Interview, Tracker) + hamburger menu for secondary screens (Knowledge, Jobs, Profile)
- **Transitions**: Slide from right for forward navigation; slide from left for back navigation; modal for create/edit overlays

### Keyboard Shortcuts (Desktop)

| Shortcut | Action |
|----------|--------|
| `Ctrl + Enter` | Send message in Chat / Submit answer in Interview |
| `Ctrl + K` | Focus search bar (Knowledge) or global search |
| `Ctrl + N` | New application (Tracker) / New chat (Chat) |
| `Escape` | Close modal / sidebar |

---

## Mock Data Requirements

All mockups must use realistic placeholder data. Below are the mock data structures needed for each screen, matching the actual API response schemas.

### User Profile Mock

```json
{
  "user_id": 1,
  "email": "jane.smith@email.com",
  "skills": ["Python", "FastAPI", "AWS", "Docker", "Kubernetes", "PostgreSQL"],
  "experience_level": "senior",
  "target_roles": ["Senior Backend Engineer", "Staff Engineer"],
  "preferences": { "remote_ok": true, "location": "Tel Aviv" },
  "created_at": "2026-01-15T10:00:00Z"
}
```

### Resume Text Mock

```
Jane Smith
Senior Backend Engineer | jane@email.com | Tel Aviv, Israel

EXPERIENCE
Senior Backend Engineer — CyberTech Ltd. (2021–Present)
- Architected event-driven microservices processing 1M events/day using Kafka
- Led migration from AWS to hybrid cloud, reducing infrastructure costs by 35%
- Mentored team of 4 junior developers through code reviews and pair programming

Backend Developer — DataFlow Startup (2018–2021)
- Built REST APIs in Python/FastAPI handling 50K concurrent users
- Implemented CI/CD pipeline cutting deployment time by 70%

SKILLS
Python, Go, FastAPI, AWS, Docker, Kubernetes, PostgreSQL, Redis, Kafka

EDUCATION
B.Sc. Computer Science — Technion (2016)
```

### Job Description Mock

```
Senior Backend Engineer — FinTech Startup
Requirements: 5+ years Python experience, distributed systems expertise, 
AWS cloud infrastructure, payment processing knowledge, CI/CD pipelines.
Nice to have: Go, Kafka, Kubernetes experience.
```

### Resume Analysis Response Mock

```json
{
  "overall_score": 7.5,
  "strengths": [
    "Strong quantified achievements with metrics",
    "Relevant technical skills for backend roles",
    "Clear progression from mid to senior"
  ],
  "weaknesses": [
    "Missing professional summary section",
    "No certifications or continued education",
    "Limited mention of payment processing experience"
  ],
  "recommendations": [
    "Add a 2-3 sentence professional summary",
    "Include relevant certifications (AWS, etc.)",
    "Add payment processing projects or coursework"
  ],
  "ats_compatibility": {
    "score": 8.0,
    "issues": [],
    "suggestions": ["Add more role-specific keywords like 'payment processing'"]
  },
  "keyword_analysis": {
    "present_keywords": ["Python", "AWS", "Docker", "Kubernetes", "CI/CD"],
    "missing_keywords": ["payment processing", "Terraform", "microservices architecture"],
    "keyword_density_notes": "Good keyword coverage for backend roles"
  },
  "section_feedback": {
    "contact_info": "Complete and well-formatted",
    "summary": "Missing — strongly recommended to add",
    "experience": "Strong with quantified metrics",
    "skills": "Well-organized and relevant",
    "education": "Adequate"
  },
  "processing_time": 3.42
}
```

### Interview Session Mocks

```json
{
  "session_id": "intv-a1b2c3d4-5678",
  "role": "Senior Backend Engineer",
  "level": "senior",
  "first_question": {
    "id": "q1",
    "question": "Tell me about a time you led a complex migration project. What was your approach and what were the results?",
    "type": "behavioral",
    "difficulty": "hard",
    "key_points": ["Leadership", "Technical complexity", "Measurable outcome"]
  },
  "total_questions": 5
}
```

```json
{
  "feedback": {
    "overall_score": 7.5,
    "strength_areas": ["Clear STAR structure", "Specific metrics and outcomes"],
    "improvement_areas": ["Could elaborate on team dynamics", "Missing stakeholder communication details"],
    "specific_feedback": "Good use of quantified results. The migration scope was clear.",
    "suggested_improvement": "Consider adding more context about how you managed cross-team dependencies and communicated progress to stakeholders."
  },
  "next_question": {
    "id": "q2",
    "question": "How would you design a rate-limiting system for a high-traffic API?",
    "type": "technical",
    "difficulty": "hard",
    "key_points": ["System design", "Scalability", "Trade-offs"]
  },
  "session_complete": false,
  "session_summary": null
}
```

### Knowledge Q&A Response Mock

```json
{
  "answer": "When negotiating salary, start by researching market rates using resources like Glassdoor, Levels.fyi, and industry surveys. Key strategies include: (1) Let the employer make the first offer, (2) Counter with data-backed numbers, (3) Negotiate the total compensation package including equity, bonuses, and benefits, (4) Practice your negotiation pitch beforehand.",
  "sources": ["salary_negotiation.md", "industry_insights.md"],
  "relevance_score": 0.87,
  "related_topics": ["benefits negotiation", "equity packages", "counter-offers", "market rate research"]
}
```

### Job Search Response Mock

```json
{
  "jobs": [
    {
      "title": "Senior Python Developer",
      "company": "TechCorp Israel",
      "location": "Tel Aviv, Israel",
      "description": "Build scalable backend services for our fintech platform...",
      "url": null,
      "salary_range": "$120k–$160k",
      "remote_friendly": true,
      "match_score": 0.92,
      "experience_level": "senior",
      "key_skills": ["Python", "FastAPI", "AWS", "PostgreSQL"]
    },
    {
      "title": "Backend Team Lead",
      "company": "FinStart",
      "location": "Tel Aviv, Israel",
      "description": "Lead a team of 6 engineers building payment infrastructure...",
      "url": null,
      "salary_range": "$140k–$180k",
      "remote_friendly": false,
      "match_score": 0.78,
      "experience_level": "lead",
      "key_skills": ["Go", "Python", "Kubernetes", "Payment Systems"]
    }
  ],
  "total_found": 5,
  "search_query": "Python backend developer",
  "location": "Tel Aviv",
  "location_info": {
    "city": "Tel Aviv",
    "lat": 32.0853,
    "lon": 34.7818,
    "display_name": "Tel Aviv-Yafo, Israel",
    "country": "Israel",
    "found": true
  },
  "nearby_companies": [
    { "name": "Waze", "type": "technology", "distance_m": 1200, "address": "Yigal Alon St." },
    { "name": "Monday.com", "type": "technology", "distance_m": 800, "address": "Dov Hoz St." }
  ],
  "processing_time": 2.8
}
```

### Application Tracker Mocks

```json
[
  {
    "id": 1,
    "company_name": "Google",
    "position_title": "Senior Backend Engineer",
    "job_url": "https://careers.google.com/jobs/12345",
    "status": "applied",
    "application_date": "2026-03-01",
    "follow_up_date": "2026-03-10",
    "notes": "Referred by Alice from the platform team",
    "created_at": "2026-03-01T10:00:00Z",
    "updated_at": "2026-03-01T10:00:00Z"
  },
  {
    "id": 2,
    "company_name": "Meta",
    "position_title": "Production Engineer",
    "job_url": "https://metacareers.com/jobs/67890",
    "status": "interviewing",
    "application_date": "2026-02-25",
    "follow_up_date": "2026-03-12",
    "notes": "Phone screen completed. On-site scheduled.",
    "created_at": "2026-02-25T14:00:00Z",
    "updated_at": "2026-03-03T09:00:00Z"
  },
  {
    "id": 3,
    "company_name": "Amazon",
    "position_title": "SDE II",
    "job_url": "https://amazon.jobs/en/jobs/54321",
    "status": "applied",
    "application_date": "2026-03-04",
    "follow_up_date": null,
    "notes": null,
    "created_at": "2026-03-04T16:30:00Z",
    "updated_at": "2026-03-04T16:30:00Z"
  }
]
```

### User Insights Response Mock

```json
{
  "insights": [
    "You frequently ask about resume formatting and ATS optimization",
    "Interview practice sessions have been increasing over the past 2 weeks",
    "Most job searches focus on backend engineering roles in Israel"
  ],
  "patterns": [
    "Peak activity on weekday evenings (18:00–22:00)",
    "Resume improvements followed by job search sessions",
    "Interview practice tends to follow application submissions"
  ],
  "recommendations": [
    "Consider focusing on system design preparation — it appeared as a weakness in recent interviews",
    "Add cloud certifications (AWS Solutions Architect) to strengthen your resume",
    "Diversify job search to include remote-first EU companies"
  ],
  "conversation_count": 15,
  "agent_usage": {
    "resume": 8,
    "interview": 5,
    "knowledge": 2,
    "job_search": 3,
    "memory": 15,
    "router": 15
  }
}
```

### Async Task Mocks

```json
{
  "task_id": "task-resume-audit-abc123",
  "status": "queued",
  "message": "Resume audit queued for processing",
  "estimated_completion": "2026-03-05T20:05:00Z"
}
```

```json
{
  "task_id": "task-resume-audit-abc123",
  "status": "completed",
  "result": {
    "detailed_analysis": "...",
    "skill_gap_report": "...",
    "industry_benchmarking": "..."
  }
}
```

---

## RTL Considerations

The current system is **English-only** (as noted in the project plan under "Functional Limitations"). However, since the project targets users in Israel and mentions Tel Aviv as a primary location, RTL support should be planned for:

### Current State
- All API responses are in English
- All RAG knowledge documents are in English
- UI text direction: LTR

### Future RTL Readiness

If Hebrew or Arabic localization is added in the future, the following design decisions must hold:

| Area | Guideline |
|------|-----------|
| **CSS Direction** | Use `direction: rtl` and `dir="rtl"` on `<html>` when locale is Hebrew/Arabic. Use logical CSS properties (`margin-inline-start` instead of `margin-left`, `padding-inline-end` instead of `padding-right`). |
| **Layout Mirroring** | Sidebar moves to the right. Kanban columns flow right-to-left. Chat bubbles: user on left, AI on right (reversed from LTR). |
| **Icons** | Directional icons (arrows, chevrons, expand) must flip. Non-directional icons (search, settings, delete) remain unchanged. |
| **Text Alignment** | Use `text-align: start` (not `left`). Mixed-language content (English code snippets inside Hebrew text) uses Unicode BiDi algorithm — avoid manual `dir` overrides on inline code. |
| **Form Inputs** | Labels align to the right. Input fields maintain natural text direction (Hebrew text flows RTL; English/code inputs remain LTR within the field). |
| **Numbers & Dates** | Scores and percentages remain LTR. Dates should follow locale format. |
| **Component Library** | Choose a UI framework with built-in RTL support (e.g., MUI with `theme.direction`, Ant Design with `ConfigProvider direction`). |

### Mockup Implication

- **For this iteration**: Design all mockups in LTR only.
- **Spacing and alignment**: Use logical spacing tokens (start/end) in design specs so developers can implement with logical CSS properties from day one.
- **Font choice**: Select a font family that supports both Latin and Hebrew character sets (e.g., Noto Sans, Inter + Noto Sans Hebrew).

---

## Next Steps Before Implementation

### Phase 1: Mockup Creation (Weeks 1–2)

| # | Task | Tool | Deliverable |
|---|------|------|-------------|
| 1 | Set up design system (colors, typography, spacing tokens, component library) | Figma / Sketch | Design system file |
| 2 | Create wireframes for all 9 screens (low-fidelity) | Figma / Excalidraw | Wireframe deck |
| 3 | Build high-fidelity mockups for P0 screens (Dashboard, Chat, Resume Studio, Interview, Tracker) | Figma | HiFi mockups |
| 4 | Build high-fidelity mockups for P1 screens (Knowledge, Job Search, Profile, Async Tasks) | Figma | HiFi mockups |
| 5 | Create interactive prototype linking all screens | Figma / InVision | Clickable prototype |
| 6 | Populate all mockups with mock data from this document | — | Data-realistic screens |
| 7 | Design responsive variants (mobile breakpoint: 768px) | Figma | Mobile mockups |

### Phase 2: Review & Validation (Week 3)

| # | Task | Participants | Deliverable |
|---|------|-------------|-------------|
| 1 | Walkthrough with backend engineer to verify endpoint coverage | Backend dev | Coverage checklist |
| 2 | UX review for interaction consistency | UX reviewer | Feedback document |
| 3 | Accessibility audit (contrast ratios, focus indicators, screen reader labels) | Designer | A11y report |
| 4 | Finalize component specs (spacing, sizing, breakpoints) for developer handoff | Designer | Component spec sheet |

### Phase 3: Implementation Prep (Week 4)

| # | Task | Deliverable |
|---|------|-------------|
| 1 | Export design tokens (colors, typography, spacing) as JSON/CSS variables | `tokens.json` |
| 2 | Create component inventory with prop definitions matching this document | Component checklist |
| 3 | Set up frontend project scaffold (framework, routing, API client) | Repo / boilerplate |
| 4 | Configure API client with base URL, `X-API-Key` header, and error interceptor | API service module |
| 5 | Implement mock data service (using mock JSONs from this document) for offline development | Mock service |

### Design System Recommendations

| Token | Value | Notes |
|-------|-------|-------|
| Primary Color | `#2563EB` (Blue 600) | Buttons, links, active states |
| Success | `#16A34A` (Green 600) | Strengths, connected, scores ≥ 7 |
| Warning | `#D97706` (Amber 600) | Follow-up due, scores 4–6 |
| Danger | `#DC2626` (Red 600) | Weaknesses, errors, scores < 4, overdue |
| Neutral | `#6B7280` (Gray 500) | Secondary text, borders |
| Background | `#F9FAFB` (Gray 50) | Page background |
| Surface | `#FFFFFF` | Card backgrounds |
| Font Family | `Inter, system-ui, sans-serif` | Body text |
| Font Mono | `JetBrains Mono, monospace` | Code, scores, IDs |
| Border Radius | `8px` (cards), `6px` (buttons), `4px` (inputs) | Consistent rounding |
| Spacing Scale | `4, 8, 12, 16, 24, 32, 48, 64` | 4px base grid |
| Breakpoints | `640px` (sm), `768px` (md), `1024px` (lg), `1280px` (xl) | Tailwind defaults |

---

## Appendix: Endpoint-to-Component Traceability Matrix

This matrix ensures every API endpoint has at least one UI component that invokes it:

| Endpoint | Component(s) | Screen(s) | Trigger |
|----------|-------------|-----------|---------|
| `GET /health` | `HealthIndicator` | TopNav (all) | Auto-poll every 30s |
| `POST /chat` | `ChatBubble`, `AttachmentBar` | Chat | Send button / Ctrl+Enter |
| `POST /resume` | Resume Input Panel → `ScoreGauge`, `StrengthWeaknessList`, `ATSPanel` | Resume Studio, Chat | "Analyze" button |
| `POST /resume/improve` | `BulletDiff`, Priority Actions Bar | Resume Studio, Chat | "Improve" button |
| `POST /resume/audit` | `AsyncTaskBanner` | Resume Studio | "Full Audit" button |
| `POST /interview/start` | Setup Form → `InterviewQuestionCard` | Interview Practice, Chat | "Start Interview" button |
| `POST /interview/answer` | Answer Textarea → `InterviewFeedbackCard` | Interview Practice, Chat | "Submit Answer" button |
| `GET /interview/questions/{job_title}` | Question List | Interview Practice | Quick-questions tab |
| `POST /interview/report` | `AsyncTaskBanner` | Interview Practice | "Generate Report" button |
| `POST /ask` | Search Bar → `KnowledgeAnswerCard` | Knowledge Base, Chat | "Ask" button |
| `POST /jobs/search` | Search Form → `JobCard` grid | Job Search, Chat | "Search" button |
| `POST /jobs/match` | `JobCard` (match score overlay) | Job Search | "Match Resume" button |
| `GET /jobs/location/{city}` | `LocationInfoCard`, `NearbyCompanyList` | Job Search | Auto on search |
| `POST /applications` | `ApplicationFormModal` | Tracker, Job Search | "New Application" / "Save to Tracker" |
| `GET /applications` | `KanbanColumn`, `ApplicationCard` | Tracker | Auto on page load |
| `PUT /applications/{id}` | `ApplicationFormModal`, drag-and-drop | Tracker | Edit / drag card |
| `DELETE /applications/{id}` | `ConfirmDialog` | Tracker | Delete button |
| `GET /applications/follow-ups` | `FollowUpBadge`, Follow-ups Widget | Tracker, Dashboard | Auto on load / filter |
| `POST /applications/batch-update` | Batch Action Bar | Tracker | "Batch Update" button |
| `GET /user/{id}/profile` | Profile Form | Profile & Insights | Auto on tab load |
| `POST /user/{id}/profile/update` | Profile Form | Profile & Insights | "Save" button |
| `GET /user/{id}/context` | History List, Context Summary | Profile & Insights, Dashboard | Auto on tab/page load |
| `GET /user/{id}/insights` | Insights List, `AgentUsageChart` | Profile & Insights, Dashboard | Auto on tab/page load |
| `GET /tasks/{id}/status` | `AsyncTaskBanner` | Resume Studio, Interview | Poll every 5s while task active |
| `GET /result/{id}` | Result Viewer | Resume Studio, Interview | Auto when task completes |
