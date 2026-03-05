# Frontend Implementation Plan

## Overview

This document defines the step-by-step frontend implementation strategy for the AI Job Application Coach. The frontend provides a rich, interactive web interface to consume the existing FastAPI backend (25 REST endpoints, 6 LLM-powered agents, LangGraph orchestration, RAG-based knowledge retrieval, async task processing, and user memory).

### Goals

| # | Goal | Success Criteria |
|---|------|-----------------|
| 1 | Unified conversational interface for all 6 agents | Users can chat naturally; routing is transparent |
| 2 | Structured task-specific UIs for power users | Dedicated pages for resume analysis, interview practice, job search, and application tracking |
| 3 | Real-time feedback on long-running operations | Polling-based progress for async tasks (resume audit, interview report, batch updates) |
| 4 | Responsive, accessible, mobile-friendly design | WCAG 2.1 AA compliance; usable on screens ≥ 360 px |
| 5 | Production-grade error handling and loading states | No unhandled promise rejections; skeleton loaders for all async data |

### Tech Stack Decision

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Framework | **React 18** (TypeScript) | Component model, ecosystem maturity, team familiarity |
| Routing | **React Router v6** | Nested layouts, route-based code splitting |
| State Management | **Zustand** + **TanStack Query (React Query v5)** | Zustand for client state; TanStack Query for server state, caching, polling |
| Styling | **Tailwind CSS 3** + **shadcn/ui** | Utility-first CSS; accessible, composable component primitives |
| HTTP Client | **Axios** (with interceptors) | Request/response interceptors for auth, error normalization, retry |
| Forms | **React Hook Form** + **Zod** | Type-safe validation mirroring backend Pydantic schemas |
| Markdown Rendering | **react-markdown** + **remark-gfm** | Render LLM-generated Markdown responses |
| Charts | **Recharts** | Lightweight charting for score visualizations and analytics |
| Testing | **Vitest** + **React Testing Library** + **Playwright** | Unit, integration, and E2E testing |
| Build | **Vite 5** | Fast HMR, ESBuild bundling, optimized production builds |
| Linting | **ESLint** + **Prettier** + **typescript-eslint** | Code quality and formatting consistency |

### Estimated Timeline

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| Phase A: Foundation & Layout | Week 1–2 | Project scaffold, API client, auth, shell layout, health dashboard |
| Phase B: Core Features | Week 3–5 | Chat, resume, interview, knowledge, job search pages |
| Phase C: Application Tracker | Week 6 | CRUD table, status workflow, follow-up reminders |
| Phase D: User Profile & Memory | Week 7 | Profile page, conversation history, AI insights |
| Phase E: Polish & Testing | Week 8–9 | Error boundaries, loading skeletons, full test suite, accessibility audit |
| Phase F: Performance & Deployment | Week 10 | Bundle optimization, CI/CD, Docker integration |

---

## Frontend Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Browser (SPA)                              │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    React Router v6                             │  │
│  │  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────────┐ │  │
│  │  │  Chat   │ │  Resume  │ │ Interview │ │  Applications    │ │  │
│  │  │  Page   │ │  Page    │ │   Page    │ │     Page         │ │  │
│  │  └────┬────┘ └────┬─────┘ └────┬──────┘ └────┬─────────────┘ │  │
│  │       │           │            │              │               │  │
│  │  ┌────▼───────────▼────────────▼──────────────▼────────────┐  │  │
│  │  │              Shared Component Library                    │  │  │
│  │  │  (ScoreCard, ChatBubble, StatusBadge, LoadingSkeleton)  │  │  │
│  │  └─────────────────────┬───────────────────────────────────┘  │  │
│  └────────────────────────│──────────────────────────────────────┘  │
│                           │                                         │
│  ┌────────────────────────▼──────────────────────────────────────┐  │
│  │                 State Layer                                    │  │
│  │  ┌──────────────────┐  ┌──────────────────────────────────┐   │  │
│  │  │  Zustand Stores  │  │  TanStack Query Cache            │   │  │
│  │  │  (auth, UI,      │  │  (server data: resume results,   │   │  │
│  │  │   active session)│  │   applications, user profile)    │   │  │
│  │  └──────────────────┘  └──────────────────────────────────┘   │  │
│  └────────────────────────┬──────────────────────────────────────┘  │
│                           │                                         │
│  ┌────────────────────────▼──────────────────────────────────────┐  │
│  │               API Client (Axios)                               │  │
│  │  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌────────────┐  │  │
│  │  │Auth      │  │Rate Limit  │  │Retry     │  │Error       │  │  │
│  │  │Intercept.│  │Handler     │  │Logic     │  │Normalizer  │  │  │
│  │  └──────────┘  └────────────┘  └──────────┘  └────────────┘  │  │
│  └────────────────────────┬──────────────────────────────────────┘  │
│                           │                                         │
└───────────────────────────│─────────────────────────────────────────┘
                            │ HTTPS / JSON
┌───────────────────────────▼─────────────────────────────────────────┐
│               FastAPI Backend (http://localhost:8000)                │
│   25 REST endpoints · 6 agents · LangGraph · Celery · MySQL · RAG  │
└─────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
frontend/
├── public/
│   ├── favicon.ico
│   └── manifest.json
├── src/
│   ├── main.tsx                     # React entry point
│   ├── App.tsx                      # Router + providers
│   ├── vite-env.d.ts
│   │
│   ├── api/                         # API client layer
│   │   ├── client.ts                # Axios instance + interceptors
│   │   ├── endpoints.ts             # Typed endpoint functions
│   │   ├── types.ts                 # Request/response TypeScript interfaces
│   │   └── polling.ts               # Async task polling utilities
│   │
│   ├── stores/                      # Zustand stores (client state)
│   │   ├── authStore.ts             # API key, user ID
│   │   ├── uiStore.ts              # Sidebar, theme, toasts
│   │   ├── chatStore.ts             # Chat message history (local)
│   │   └── interviewStore.ts        # Active interview session state
│   │
│   ├── hooks/                       # Custom React hooks
│   │   ├── useResumeAnalysis.ts     # TanStack Query wrapper for /resume
│   │   ├── useInterview.ts          # Interview session management
│   │   ├── useApplications.ts       # Application CRUD + optimistic updates
│   │   ├── useChat.ts               # Chat mutation + message append
│   │   ├── useJobSearch.ts          # Job search queries
│   │   ├── useKnowledge.ts          # Knowledge Q&A queries
│   │   ├── useUserProfile.ts        # User profile + insights
│   │   └── useAsyncTask.ts          # Generic async task polling hook
│   │
│   ├── pages/                       # Route-level page components
│   │   ├── DashboardPage.tsx        # Landing / overview dashboard
│   │   ├── ChatPage.tsx             # Unified conversational interface
│   │   ├── ResumePage.tsx           # Resume analysis + improvement
│   │   ├── InterviewPage.tsx        # Mock interview practice
│   │   ├── JobSearchPage.tsx        # Job search + matching
│   │   ├── KnowledgePage.tsx        # Career Q&A
│   │   ├── ApplicationsPage.tsx     # Application tracker (table)
│   │   ├── UserProfilePage.tsx      # Memory / insights / history
│   │   ├── SettingsPage.tsx         # API key, preferences
│   │   └── NotFoundPage.tsx         # 404
│   │
│   ├── components/                  # Shared UI components
│   │   ├── layout/
│   │   │   ├── AppShell.tsx         # Sidebar + topbar + main area
│   │   │   ├── Sidebar.tsx          # Navigation links
│   │   │   └── TopBar.tsx           # User info, health indicator
│   │   ├── chat/
│   │   │   ├── ChatWindow.tsx       # Scrollable message list
│   │   │   ├── ChatBubble.tsx       # Single message (user / AI)
│   │   │   ├── ChatInput.tsx        # Text input + attachment triggers
│   │   │   └── TypingIndicator.tsx  # AI "thinking" animation
│   │   ├── resume/
│   │   │   ├── ResumeUpload.tsx     # Text area / file upload
│   │   │   ├── ScoreCard.tsx        # Circular score gauge
│   │   │   ├── StrengthsList.tsx    # Strengths / weaknesses lists
│   │   │   ├── ATSReport.tsx        # ATS compatibility panel
│   │   │   ├── KeywordAnalysis.tsx  # Present / missing keywords
│   │   │   ├── SectionFeedback.tsx  # Per-section feedback cards
│   │   │   └── ImprovementDiff.tsx  # Original vs improved bullet comparison
│   │   ├── interview/
│   │   │   ├── SessionSetup.tsx     # Role, level, question count form
│   │   │   ├── QuestionCard.tsx     # Current question display
│   │   │   ├── AnswerInput.tsx      # Answer text area + submit
│   │   │   ├── FeedbackPanel.tsx    # STAR evaluation scores + feedback
│   │   │   ├── SessionSummary.tsx   # Aggregate performance report
│   │   │   └── ProgressBar.tsx      # Question N of M progress
│   │   ├── jobs/
│   │   │   ├── SearchForm.tsx       # Query, location, level, remote toggle
│   │   │   ├── JobCard.tsx          # Single job listing
│   │   │   ├── LocationMap.tsx      # Geocoded location display
│   │   │   └── MatchScore.tsx       # Profile–job match gauge
│   │   ├── applications/
│   │   │   ├── ApplicationTable.tsx # Sortable, filterable data table
│   │   │   ├── ApplicationForm.tsx  # Create / edit form
│   │   │   ├── StatusBadge.tsx      # Color-coded status pill
│   │   │   ├── StatusWorkflow.tsx   # Visual status transition diagram
│   │   │   └── FollowUpAlert.tsx    # Upcoming follow-up reminders
│   │   ├── user/
│   │   │   ├── ProfileCard.tsx      # Skills, experience level, goals
│   │   │   ├── InsightsPanel.tsx    # AI-generated conversation patterns
│   │   │   └── AgentUsageChart.tsx  # Agent usage distribution (Recharts)
│   │   └── shared/
│   │       ├── LoadingSkeleton.tsx   # Shimmer placeholder
│   │       ├── ErrorBoundary.tsx     # React error boundary
│   │       ├── ErrorAlert.tsx        # Inline error display
│   │       ├── EmptyState.tsx        # No-data illustration
│   │       ├── ConfirmDialog.tsx     # Destructive action confirmation
│   │       ├── Toast.tsx             # Notification toast
│   │       ├── AsyncTaskBanner.tsx   # Task polling progress bar
│   │       └── HealthIndicator.tsx   # Green/yellow/red dot
│   │
│   ├── lib/                         # Utility functions
│   │   ├── constants.ts             # API base URL, routes, status colors
│   │   ├── formatters.ts            # Date, score, duration formatting
│   │   ├── validators.ts            # Zod schemas matching Pydantic models
│   │   └── cn.ts                    # Tailwind className merge utility
│   │
│   └── styles/
│       └── globals.css              # Tailwind directives + custom tokens
│
├── tests/
│   ├── unit/                        # Vitest component + hook tests
│   ├── integration/                 # Multi-component interaction tests
│   └── e2e/                         # Playwright browser tests
│
├── .env.example                     # VITE_API_BASE_URL, VITE_API_KEY
├── .eslintrc.cjs
├── .prettierrc
├── tailwind.config.ts
├── tsconfig.json
├── tsconfig.app.json
├── vite.config.ts
├── playwright.config.ts
├── package.json
└── README.md
```

### Routing Map

| Path | Page Component | Backend Endpoints Used |
|------|---------------|----------------------|
| `/` | `DashboardPage` | `GET /health`, `GET /user/{id}/context`, `GET /applications/follow-ups` |
| `/chat` | `ChatPage` | `POST /chat` |
| `/resume` | `ResumePage` | `POST /resume`, `POST /resume/improve`, `POST /resume/audit`, `GET /tasks/{id}/status` |
| `/interview` | `InterviewPage` | `POST /interview/start`, `POST /interview/answer`, `GET /interview/questions/{title}`, `POST /interview/report` |
| `/jobs` | `JobSearchPage` | `POST /jobs/search`, `POST /jobs/match`, `GET /jobs/location/{city}` |
| `/knowledge` | `KnowledgePage` | `POST /ask` |
| `/applications` | `ApplicationsPage` | `POST /applications`, `GET /applications`, `PUT /applications/{id}`, `DELETE /applications/{id}`, `GET /applications/follow-ups`, `POST /applications/batch-update` |
| `/profile` | `UserProfilePage` | `GET /user/{id}/profile`, `GET /user/{id}/context`, `GET /user/{id}/insights`, `POST /user/{id}/profile/update` |
| `/settings` | `SettingsPage` | `GET /health` |
| `*` | `NotFoundPage` | — |

---

## Pages & Components

### Dashboard Page

The landing page provides a high-level overview of the user's career coaching activity.

**Sections:**
1. **Health Indicator** — Live status dot (green/yellow/red) driven by `GET /health`. Shows database, Redis, and ChromaDB connectivity.
2. **Quick Actions** — Card grid linking to: "Analyse Resume", "Practice Interview", "Search Jobs", "Ask a Question".
3. **Recent Activity** — Last 5 conversations from `GET /user/{id}/context` showing intent, summary, and timestamp.
4. **Follow-Up Reminders** — Upcoming follow-up dates from `GET /applications/follow-ups` displayed as alert cards.
5. **Agent Usage Chart** — Pie or bar chart from `GET /user/{id}/insights` → `agent_usage` showing distribution across agents.

### Chat Page

The primary conversational interface consuming `POST /chat`. This is the "smart" entry point that auto-routes to the right agent.

**Layout:**
- Left panel (optional, collapsible): Conversation history list.
- Center: Scrollable chat message area (`ChatWindow`).
- Bottom: Multi-line input (`ChatInput`) with optional attachment triggers (resume text, job description, interview session ID).

**Behavior:**
1. User types a message → `POST /chat` with `{ message, user_id, ...optional fields }`.
2. While awaiting response: show `TypingIndicator` in the message area.
3. On response: render `ChatBubble` with the AI response. If `data` contains structured output (e.g., resume scores), render an inline card (e.g., `ScoreCard`).
4. Display `intent`, `confidence`, `agents_used`, and `processing_time` as metadata below the AI message.
5. If the response contains `interview_session_id`, persist it in `interviewStore` so follow-up messages include it.

**Data Fields Sent to `/chat`:**

| Field | Source |
|-------|--------|
| `message` | Chat input text |
| `user_id` | `authStore.userId` |
| `session_id` | Auto-generated UUID per conversation |
| `resume_text` | Attached via "Attach Resume" button |
| `job_description` | Attached via "Attach Job Description" button |
| `interview_role` | Set when user starts an interview via chat |
| `interview_level` | Set when user starts an interview via chat |
| `interview_session_id` | From `interviewStore` if active |
| `interview_answer` | Detected when replying to an interview question |
| `interview_question_id` | From the last received question |
| `job_search_location` | Extracted or set via attachment |

### Resume Page

Dedicated structured interface for resume analysis and improvement.

**Sections:**
1. **Input Form** — `ResumeUpload` component:
   - Large text area for resume text (min 50 chars).
   - Optional text area for job description.
   - "Analyse" button → `POST /resume`.
   - "Improve" button → `POST /resume/improve`.
   - "Full Audit" button → `POST /resume/audit` (async).
2. **Analysis Results** (shown after `/resume` response):
   - `ScoreCard` — Overall score (1–10) as a circular gauge.
   - `StrengthsList` — Green-tinted list of strengths.
   - `StrengthsList` (weakness variant) — Red-tinted list of weaknesses.
   - `ATSReport` — ATS score + issues + suggestions.
   - `KeywordAnalysis` — Two-column layout: present keywords (tagged green) vs missing keywords (tagged red).
   - `SectionFeedback` — Accordion or card per section (contact, summary, experience, skills, education).
3. **Improvement Results** (shown after `/resume/improve` response):
   - `ImprovementDiff` — Side-by-side original vs improved bullets with reasoning.
   - Improved summary text.
   - Priority actions as a numbered checklist.
4. **Async Audit Banner** — When a full audit is queued, `AsyncTaskBanner` polls `GET /tasks/{id}/status` and displays progress. On `SUCCESS`, renders the full audit report.

### Interview Page

Multi-step mock interview practice interface.

**States:**
1. **Setup** (`SessionSetup`):
   - Form: role (text input), level (dropdown: junior/mid/senior/lead), question count (slider: 1–10).
   - "Start Interview" button → `POST /interview/start`.
   - "Quick Questions" link → `GET /interview/questions/{role}` for one-off question generation.
2. **In Progress** (after session starts):
   - `ProgressBar` — "Question 2 of 5".
   - `QuestionCard` — Displays the current question with type badge (behavioral/technical/situational) and difficulty badge.
   - `AnswerInput` — Text area (min 10 chars) + "Submit Answer" button → `POST /interview/answer`.
   - After submission: `FeedbackPanel` — Overall score (gauge), strength areas, improvement areas, specific feedback, suggested improvement.
   - "Next Question" button to proceed (next question is included in the answer response as `next_question`).
3. **Complete** (when `session_complete === true`):
   - `SessionSummary` — Aggregate scores, overall performance assessment, per-question score breakdown.
   - "Generate Full Report" button → `POST /interview/report` (async). Shows `AsyncTaskBanner` polling for completion.
   - "Start New Interview" button to reset.

### Job Search Page

Location-based job search with optional resume matching.

**Sections:**
1. **Search Form** (`SearchForm`):
   - Query (text), location (text with geocoding preview), experience level (dropdown), remote toggle, count (1–10).
   - "Search" button → `POST /jobs/search`.
   - "Match with Resume" toggle — if enabled, also sends `resume_text` → `POST /jobs/match`.
2. **Results**:
   - `JobCard` list — Each card shows: title, company, location, salary range, remote badge, key skills tags, match score (if available).
   - `LocationMap` — If `location_info.found === true`, show a simple text-based location summary (city, country, coordinates). Future: integrate Leaflet for an interactive map.
   - "Nearby Companies" section — List of nearby tech companies from `nearby_companies` response field.
3. **Empty State** — When no jobs found, show `EmptyState` with suggestions to broaden search criteria.

### Knowledge Page

RAG-powered career Q&A interface.

**Layout:**
- Single-column layout with a question input and answer display.
- Previously asked questions listed below for re-reference.

**Sections:**
1. **Question Input** — Text input (min 5 chars) + "Ask" button → `POST /ask`.
2. **Answer Display**:
   - Answer text rendered as Markdown (`react-markdown`).
   - Source attribution list — clickable source documents (e.g., "salary_negotiation.md").
   - Relevance score displayed as a confidence meter.
   - Related topics as clickable chips — clicking one triggers a new query for that topic.

### Applications Page

Full CRUD application tracker with status workflow management.

**Components:**
1. **Application Table** (`ApplicationTable`):
   - Columns: Company, Position, Status, Applied Date, Follow-up Date, Actions.
   - Sortable by any column; filterable by status dropdown.
   - Each row has: Edit (pencil icon), Delete (trash icon with `ConfirmDialog`).
   - Status column uses `StatusBadge` (color-coded: applied=blue, interviewing=yellow, offer=green, rejected=red, withdrawn=gray).
2. **Create / Edit Form** (`ApplicationForm`) — Modal or slide-over:
   - Fields: company name, position title, job URL, application date (date picker), notes, follow-up date.
   - For edit mode: status dropdown with only valid transitions shown (e.g., from "applied" only show "interviewing", "rejected", "withdrawn").
3. **Status Workflow Diagram** (`StatusWorkflow`):
   - Visual representation of the status state machine: `applied → interviewing → offer`, with `rejected` and `withdrawn` as terminal states reachable from all active states.
4. **Follow-Up Alerts** (`FollowUpAlert`):
   - Banner at the top showing applications with follow-up dates ≤ 3 days from now.
   - Links directly to the application for updating.
5. **Batch Update** — Select multiple rows → "Batch Update Status" → `POST /applications/batch-update` (async). Show `AsyncTaskBanner` during processing.

### User Profile Page

Display user memory, conversation history, and AI-generated insights.

**Sections:**
1. **Profile Card** (`ProfileCard`) — From `GET /user/{id}/profile`:
   - Skills (tag list), experience level, target roles, career goals.
   - "Edit Profile" button → `POST /user/{id}/profile/update`.
2. **Conversation Context** — From `GET /user/{id}/context`:
   - Total conversations count.
   - Top intents breakdown.
   - Recent conversations list (intent, summary, timestamp).
3. **AI Insights** (`InsightsPanel`) — From `GET /user/{id}/insights`:
   - Insight text items (e.g., "User frequently asks about resume formatting").
   - Detected patterns (e.g., "Interview practice sessions increasing").
   - AI recommendations (e.g., "Consider focusing on system design preparation").
4. **Agent Usage Chart** (`AgentUsageChart`) — Pie chart of `agent_usage` breakdown.

### Settings Page

Configuration and preferences.

- **API Key** — Input field to set/update the `X-API-Key` value stored in `authStore`. Saved to `localStorage`.
- **User ID** — Configurable user ID (default 1). Persisted in `authStore`.
- **API Base URL** — Configurable for connecting to different environments (localhost, staging, production).
- **Health Check** — Live system health display from `GET /health` with per-dependency status.

---

## Data Flow & API Integration

### API Client Architecture

```typescript
// src/api/client.ts

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 30000, // 30s for LLM-backed endpoints
  headers: { 'Content-Type': 'application/json' },
});

// Request interceptor: attach API key + request ID
apiClient.interceptors.request.use((config) => {
  const apiKey = useAuthStore.getState().apiKey;
  if (apiKey) config.headers['X-API-Key'] = apiKey;
  config.headers['X-Request-ID'] = crypto.randomUUID();
  return config;
});

// Response interceptor: normalize errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 429) {
      // Rate limited — extract Retry-After and queue retry
    }
    if (error.response?.status === 401) {
      // Invalid API key — redirect to settings
    }
    return Promise.reject(normalizeError(error));
  }
);
```

### Endpoint Function Signatures

```typescript
// src/api/endpoints.ts

// Resume
export const analyzeResume = (data: ResumeRequest): Promise<ResumeResponse> =>
  apiClient.post('/resume', data).then(r => r.data);

export const improveResume = (data: ResumeRequest): Promise<ResumeImprovementResponse> =>
  apiClient.post('/resume/improve', data).then(r => r.data);

export const auditResume = (data: ResumeRequest): Promise<AsyncTaskResponse> =>
  apiClient.post('/resume/audit', data).then(r => r.data);

// Interview
export const startInterview = (data: InterviewStartRequest): Promise<InterviewStartResponse> =>
  apiClient.post('/interview/start', data).then(r => r.data);

export const submitAnswer = (data: InterviewAnswerRequest): Promise<InterviewAnswerResponse> =>
  apiClient.post('/interview/answer', data).then(r => r.data);

export const getQuestions = (jobTitle: string, level?: string, count?: number): Promise<InterviewQuestion[]> =>
  apiClient.get(`/interview/questions/${encodeURIComponent(jobTitle)}`, { params: { level, count } }).then(r => r.data);

export const generateReport = (data: InterviewReportRequest): Promise<AsyncTaskResponse> =>
  apiClient.post('/interview/report', data).then(r => r.data);

// Knowledge
export const askQuestion = (data: KnowledgeQueryRequest): Promise<KnowledgeQueryResponse> =>
  apiClient.post('/ask', data).then(r => r.data);

// Jobs
export const searchJobs = (data: JobSearchRequest): Promise<JobSearchResponse> =>
  apiClient.post('/jobs/search', data).then(r => r.data);

export const matchJobs = (data: JobMatchRequest): Promise<JobSearchResponse> =>
  apiClient.post('/jobs/match', data).then(r => r.data);

export const getLocationInfo = (city: string): Promise<LocationInfo> =>
  apiClient.get(`/jobs/location/${encodeURIComponent(city)}`).then(r => r.data);

// Applications
export const createApplication = (data: ApplicationCreateRequest): Promise<ApplicationResponse> =>
  apiClient.post('/applications', data).then(r => r.data);

export const listApplications = (userId: number, status?: string): Promise<ApplicationResponse[]> =>
  apiClient.get('/applications', { params: { user_id: userId, status } }).then(r => r.data);

export const updateApplication = (id: number, data: ApplicationUpdateRequest): Promise<ApplicationResponse> =>
  apiClient.put(`/applications/${id}`, data).then(r => r.data);

export const deleteApplication = (id: number): Promise<void> =>
  apiClient.delete(`/applications/${id}`);

export const getFollowUps = (userId: number): Promise<ApplicationResponse[]> =>
  apiClient.get('/applications/follow-ups', { params: { user_id: userId } }).then(r => r.data);

export const batchUpdateApplications = (data: BatchUpdateRequest): Promise<AsyncTaskResponse> =>
  apiClient.post('/applications/batch-update', data).then(r => r.data);

// User / Memory
export const getUserProfile = (userId: number): Promise<UserProfile> =>
  apiClient.get(`/user/${userId}/profile`).then(r => r.data);

export const getUserContext = (userId: number): Promise<UserContextResponse> =>
  apiClient.get(`/user/${userId}/context`).then(r => r.data);

export const getUserInsights = (userId: number): Promise<ConversationAnalysisResponse> =>
  apiClient.get(`/user/${userId}/insights`).then(r => r.data);

export const updateUserProfile = (userId: number, data: ProfileUpdateRequest): Promise<UserProfile> =>
  apiClient.post(`/user/${userId}/profile/update`, data).then(r => r.data);

// Chat
export const sendChatMessage = (data: ChatRequest): Promise<ChatResponse> =>
  apiClient.post('/chat', data).then(r => r.data);

// Tasks
export const getTaskStatus = (taskId: string): Promise<TaskStatusResponse> =>
  apiClient.get(`/tasks/${taskId}/status`).then(r => r.data);

export const getTaskResult = (taskId: string): Promise<any> =>
  apiClient.get(`/result/${taskId}`).then(r => r.data);

// Health
export const getHealth = (): Promise<HealthResponse> =>
  apiClient.get('/health').then(r => r.data);
```

### Async Task Polling Pattern

For endpoints that return `202 Accepted` with a `task_id` (resume audit, interview report, batch update), the frontend uses a polling hook:

```typescript
// src/hooks/useAsyncTask.ts

export function useAsyncTask(taskId: string | null) {
  return useQuery({
    queryKey: ['task', taskId],
    queryFn: () => getTaskStatus(taskId!),
    enabled: !!taskId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === 'SUCCESS' || status === 'FAILURE') return false; // Stop polling
      return 2000; // Poll every 2 seconds
    },
  });
}
```

**Polling lifecycle:**
1. User triggers async action → receives `task_id` + `status: "queued"`.
2. `useAsyncTask(taskId)` begins polling `GET /tasks/{task_id}/status` every 2 seconds.
3. `AsyncTaskBanner` renders: progress bar (indeterminate during `PENDING`/`STARTED`), status text.
4. When `status === "SUCCESS"` → polling stops → fetch `GET /result/{task_id}` → render full result.
5. When `status === "FAILURE"` → polling stops → display `ErrorAlert` with failure details.

### Data Flow Diagrams

#### Resume Analysis Flow

```
User fills form          POST /resume                Backend processes
┌─────────────┐         ┌──────────────┐            ┌──────────────┐
│ ResumeUpload │───────▶│ useResume    │───────────▶│  /resume     │
│   Component  │        │ Analysis()   │            │  endpoint    │
└─────────────┘         └──────┬───────┘            └──────┬───────┘
                               │                           │
                               │ isLoading → Skeleton      │ JSON response
                               │                           │
                        ┌──────▼───────┐            ┌──────▼───────┐
                        │ ScoreCard    │◀───────────│ ResumeAgent  │
                        │ ATSReport    │            │ analyze()    │
                        │ KeywordList  │            └──────────────┘
                        └──────────────┘
```

#### Interview Session Flow

```
Start ──▶ Q1 displayed ──▶ User answers ──▶ Feedback shown ──▶ Q2 displayed ──▶ ... ──▶ Summary

POST /interview/start   POST /interview/answer     POST /interview/answer
     │                        │                           │
     ▼                        ▼                           ▼
{ session_id,           { feedback,                 { feedback,
  first_question }        next_question }              session_complete: true,
                                                       session_summary }
```

#### Chat Flow

```
ChatInput.onSubmit()
     │
     ▼
chatStore.addMessage({ role:'user', content })
     │
     ▼
sendChatMessage({ message, user_id, ...attachments })
     │ isLoading → TypingIndicator
     ▼
chatStore.addMessage({ role:'assistant', content: response, metadata: { intent, confidence, agents_used } })
```

---

## State Management

### Zustand Stores (Client State)

Client-side state that does not come from the server. Persisted to `localStorage` where noted.

#### `authStore`

```typescript
interface AuthState {
  apiKey: string | null;         // Persisted to localStorage
  userId: number;                // Persisted to localStorage (default: 1)
  setApiKey: (key: string) => void;
  setUserId: (id: number) => void;
  clearAuth: () => void;
}
```

#### `uiStore`

```typescript
interface UIState {
  sidebarCollapsed: boolean;
  theme: 'light' | 'dark' | 'system';   // Persisted
  toasts: Toast[];
  addToast: (toast: Toast) => void;
  dismissToast: (id: string) => void;
  toggleSidebar: () => void;
}
```

#### `chatStore`

```typescript
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metadata?: {
    intent?: string;
    confidence?: number;
    agents_used?: string[];
    processing_time?: number;
    data?: Record<string, any>;   // Structured agent output
  };
  attachments?: {
    resume_text?: string;
    job_description?: string;
  };
}

interface ChatState {
  conversations: Record<string, ChatMessage[]>;  // keyed by session_id
  activeSessionId: string | null;
  addMessage: (sessionId: string, message: ChatMessage) => void;
  startNewConversation: () => string;             // Returns new session_id
  setActiveSession: (id: string) => void;
}
```

#### `interviewStore`

```typescript
interface InterviewState {
  sessionId: string | null;
  role: string;
  level: string;
  totalQuestions: number;
  currentQuestionIndex: number;
  currentQuestion: InterviewQuestion | null;
  answers: InterviewAnswerRecord[];
  sessionComplete: boolean;
  sessionSummary: SessionSummary | null;
  startSession: (response: InterviewStartResponse) => void;
  recordAnswer: (feedback: AnswerFeedback, nextQuestion?: InterviewQuestion) => void;
  completeSession: (summary: SessionSummary) => void;
  reset: () => void;
}
```

### TanStack Query (Server State)

All server-fetched data is managed by TanStack Query for automatic caching, refetching, and background updates.

**Query Key Conventions:**

| Domain | Query Key | Stale Time | Cache Time |
|--------|-----------|------------|------------|
| Health | `['health']` | 30s | 60s |
| Applications | `['applications', userId, status?]` | 60s | 5min |
| Follow-ups | `['follow-ups', userId]` | 60s | 5min |
| User Profile | `['user', userId, 'profile']` | 5min | 10min |
| User Context | `['user', userId, 'context']` | 2min | 5min |
| User Insights | `['user', userId, 'insights']` | 5min | 10min |
| Task Status | `['task', taskId]` | 0 (polling) | 30s |

**Mutation + Invalidation Rules:**

| Mutation | On Success: Invalidate |
|----------|----------------------|
| `createApplication` | `['applications']`, `['follow-ups']` |
| `updateApplication` | `['applications']`, `['follow-ups']` |
| `deleteApplication` | `['applications']`, `['follow-ups']` |
| `updateUserProfile` | `['user', userId, 'profile']`, `['user', userId, 'context']` |
| `sendChatMessage` | `['user', userId, 'context']` (memory was updated on backend) |

**Optimistic Updates:**

For `updateApplication` and `deleteApplication`, apply optimistic updates to the `['applications']` cache for instant UI feedback. Roll back on error.

```typescript
// Example: Optimistic delete
useMutation({
  mutationFn: deleteApplication,
  onMutate: async (id) => {
    await queryClient.cancelQueries({ queryKey: ['applications'] });
    const previous = queryClient.getQueryData(['applications']);
    queryClient.setQueryData(['applications'], (old) =>
      old.filter((app) => app.id !== id)
    );
    return { previous };
  },
  onError: (_err, _id, context) => {
    queryClient.setQueryData(['applications'], context.previous);
  },
  onSettled: () => {
    queryClient.invalidateQueries({ queryKey: ['applications'] });
  },
});
```

---

## Error Handling & UX States

### Error Taxonomy

The backend returns standard HTTP error codes. The frontend normalizes these into a unified error shape:

```typescript
interface AppError {
  code: number;          // HTTP status code
  message: string;       // User-friendly message
  detail?: string;       // Technical detail (from response body)
  requestId?: string;    // X-Request-ID for support/debugging
  retryable: boolean;    // Whether the user can retry
}
```

**Error Mapping:**

| HTTP Code | `message` | `retryable` | UI Action |
|-----------|-----------|-------------|-----------|
| 400 | "Invalid request. Please check your input." | false | Highlight invalid fields |
| 401 | "Authentication required. Please check your API key." | false | Redirect to Settings |
| 404 | "Resource not found." | false | Show `EmptyState` or `NotFoundPage` |
| 409 | "Invalid status transition." | false | Show toast explaining valid transitions |
| 422 | "Validation error. Please correct the highlighted fields." | false | Map field errors to form fields |
| 429 | "Rate limit reached. Please wait and try again." | true | Show countdown timer based on `Retry-After` header |
| 500 | "Something went wrong. Please try again." | true | Show `ErrorAlert` with retry button |
| Network Error | "Unable to connect to the server." | true | Show offline banner with auto-retry |

### Loading States

Every data-fetching component implements three states:

1. **Loading** — `LoadingSkeleton` matching the shape of the expected content:
   - `ScoreCard` → circular skeleton + horizontal bars.
   - `ApplicationTable` → shimmer rows.
   - `ChatBubble` → `TypingIndicator` (three animated dots).
   - `JobCard` list → 3 skeleton cards.

2. **Error** — `ErrorAlert` component:
   - Displays `message` text.
   - Shows "Retry" button if `retryable === true`.
   - Shows `requestId` in a collapsible detail section for support cases.

3. **Empty** — `EmptyState` component:
   - Illustration + contextual message (e.g., "No applications yet. Start tracking your first application.").
   - Primary action button (e.g., "Add Application").

### Global Error Boundary

```tsx
// Wraps the entire app to catch unhandled React rendering errors
<ErrorBoundary fallback={<FullPageError />}>
  <App />
</ErrorBoundary>
```

### Toast Notifications

Non-blocking notifications for background events:
- **Success**: "Application created successfully" (green, auto-dismiss 3s).
- **Warning**: "Rate limit approaching (55/60 requests used)" (yellow, auto-dismiss 5s).
- **Error**: "Failed to save application. Retry?" (red, manual dismiss with retry action).
- **Info**: "Resume audit completed! View results." (blue, clickable to navigate).

### Async Task UX

For long-running Celery tasks (resume audit, interview report, batch update):

```
┌───────────────────────────────────────────────────┐
│ ⏳ Resume audit in progress...                     │
│ ████████████░░░░░░░░░░░░░░░░░░░░  Status: STARTED │
│                                    [Cancel]        │
└───────────────────────────────────────────────────┘
         ▼ (on completion)
┌───────────────────────────────────────────────────┐
│ ✅ Resume audit complete!          [View Results]  │
└───────────────────────────────────────────────────┘
```

---

## Testing Strategy

### Testing Pyramid

```
           ┌──────────┐
          /  E2E (10)   \          Playwright: critical user journeys
         /───────────────\
        / Integration (30) \       React Testing Library: multi-component flows
       /────────────────────\
      /    Unit Tests (80+)   \    Vitest: components, hooks, stores, utilities
     /──────────────────────────\
```

### Unit Tests (Vitest + React Testing Library)

**Scope:** Individual components, hooks, stores, and utility functions in isolation.

| Target | Test File | Key Assertions |
|--------|-----------|---------------|
| `ScoreCard` | `ScoreCard.test.tsx` | Renders score value; color changes at thresholds (< 4 red, 4–7 yellow, > 7 green) |
| `StatusBadge` | `StatusBadge.test.tsx` | Correct color per status; correct label text |
| `ChatBubble` | `ChatBubble.test.tsx` | User messages right-aligned; AI messages left-aligned; Markdown rendered |
| `ImprovementDiff` | `ImprovementDiff.test.tsx` | Original and improved text displayed; reasoning visible |
| `ApplicationForm` | `ApplicationForm.test.tsx` | Validation errors shown for empty required fields; submit sends correct payload |
| `useAsyncTask` | `useAsyncTask.test.ts` | Polls at 2s intervals; stops on SUCCESS/FAILURE; returns correct status |
| `authStore` | `authStore.test.ts` | Persists apiKey to localStorage; clears on clearAuth() |
| `chatStore` | `chatStore.test.ts` | Messages append in order; new conversation generates UUID |
| `formatters` | `formatters.test.ts` | Score formatting (1 decimal), date formatting, duration formatting |
| `validators` | `validators.test.ts` | Zod schemas reject invalid payloads; accept valid payloads |
| API client interceptors | `client.test.ts` | API key header attached; errors normalized; 429 triggers retry |

**Run command:** `npx vitest` (watch mode) or `npx vitest run` (CI).

### Integration Tests (React Testing Library)

**Scope:** Multi-component interactions within a page, with mocked API responses.

| Test Suite | Scenario | Mock Endpoints |
|-----------|----------|---------------|
| `ResumePage.integration.test.tsx` | User pastes resume → clicks Analyse → sees ScoreCard + ATSReport | `POST /resume` |
| `InterviewPage.integration.test.tsx` | User starts session → answers question → sees feedback → completes session | `POST /interview/start`, `POST /interview/answer` |
| `ApplicationsPage.integration.test.tsx` | User creates application → sees in table → updates status → deletes | `POST`, `GET`, `PUT`, `DELETE /applications` |
| `ChatPage.integration.test.tsx` | User sends message → sees typing indicator → sees AI response with metadata | `POST /chat` |
| `DashboardPage.integration.test.tsx` | Dashboard loads → shows health status + follow-ups + recent activity | `GET /health`, `GET /user/{id}/context`, `GET /applications/follow-ups` |

**Mocking strategy:** Use `msw` (Mock Service Worker) to intercept HTTP requests at the network level. Define handlers matching the exact backend API contract.

```typescript
// tests/mocks/handlers.ts
import { http, HttpResponse } from 'msw';

export const handlers = [
  http.get('http://localhost:8000/health', () =>
    HttpResponse.json({ status: 'healthy', database_connected: true, checks: { database: true, redis: true, chromadb: true } })
  ),
  http.post('http://localhost:8000/resume', async ({ request }) => {
    const body = await request.json();
    return HttpResponse.json({
      overall_score: 7.5,
      strengths: ['Strong technical skills'],
      weaknesses: ['Missing summary'],
      recommendations: ['Add professional summary'],
      ats_compatibility: { score: 8.0, issues: [], suggestions: [] },
      keyword_analysis: { present_keywords: ['Python'], missing_keywords: ['Go'], keyword_density_notes: 'Good' },
      section_feedback: { contact_info: 'Complete', summary: 'Missing' },
      processing_time: 3.2,
    });
  }),
  // ... more handlers per endpoint
];
```

### End-to-End Tests (Playwright)

**Scope:** Full browser tests against a running backend (or fully mocked). Cover critical user journeys end-to-end.

| Test | Steps | Assertions |
|------|-------|-----------|
| **Happy path: Resume** | Navigate to `/resume` → paste resume + job desc → click Analyse → verify score displayed | Response renders; score is a number 1–10 |
| **Happy path: Interview** | Navigate to `/interview` → fill setup form → start → answer all questions → verify summary | Session completes; summary contains scores |
| **Happy path: Applications** | Navigate to `/applications` → create → verify in table → update status → delete → verify removed | CRUD operations reflected in UI |
| **Happy path: Chat** | Navigate to `/chat` → type "review my resume" → attach resume text → verify AI responds | Response contains intent and agents_used |
| **Auth flow** | No API key set → request fails → navigate to settings → enter key → retry → success | 401 redirects to settings; valid key succeeds |
| **Rate limiting** | Send 61 requests rapidly → verify 429 shown → wait → retry succeeds | Rate limit error shown; retry-after countdown |
| **Async task** | Trigger resume audit → verify progress banner → wait for completion → view results | Task status transitions: queued → started → success |
| **Responsive layout** | Run on 360px viewport → sidebar collapses → navigation via hamburger menu | All pages accessible on mobile |
| **Accessibility** | `axe-core` audit on each page | Zero critical a11y violations |

**Run command:** `npx playwright test` (headless) or `npx playwright test --headed` (visual).

**CI Configuration:**

```yaml
# Playwright runs in GitHub Actions
- name: Run E2E Tests
  run: npx playwright test
  env:
    VITE_API_BASE_URL: http://localhost:8000
```

---

## Performance Considerations

### Bundle Optimization

| Technique | Implementation | Expected Impact |
|-----------|---------------|----------------|
| **Code splitting** | Route-based lazy loading via `React.lazy()` + `Suspense` | Each page loaded on demand; initial bundle < 150 KB |
| **Tree shaking** | Vite production build with ESBuild | Dead code elimination for unused exports |
| **Dynamic imports** | Heavy libraries (Recharts, react-markdown) loaded lazily | Chart and Markdown rendering code excluded from initial load |
| **Asset compression** | Vite `build.rollupOptions` with gzip/brotli | 60–80% transfer size reduction |
| **Image optimization** | SVG illustrations; no raster images in core UI | Zero image download cost |

**Route-level code splitting:**

```tsx
const ChatPage = lazy(() => import('./pages/ChatPage'));
const ResumePage = lazy(() => import('./pages/ResumePage'));
const InterviewPage = lazy(() => import('./pages/InterviewPage'));
const ApplicationsPage = lazy(() => import('./pages/ApplicationsPage'));
// ... etc.

<Suspense fallback={<PageSkeleton />}>
  <Routes>
    <Route path="/chat" element={<ChatPage />} />
    <Route path="/resume" element={<ResumePage />} />
    ...
  </Routes>
</Suspense>
```

### Network Optimization

| Technique | Implementation | Rationale |
|-----------|---------------|-----------|
| **Request deduplication** | TanStack Query deduplicates identical in-flight requests | Prevents redundant API calls when components mount simultaneously |
| **Stale-while-revalidate** | TanStack Query `staleTime` configured per query | User sees cached data instantly; background refresh fetches latest |
| **Optimistic updates** | Mutations for applications CRUD | Instant UI feedback; rollback on server error |
| **Request cancellation** | Axios `CancelToken` / `AbortController` on component unmount | Prevents state updates on unmounted components |
| **Debounced inputs** | 300ms debounce on search and knowledge query inputs | Reduces unnecessary API calls during typing |
| **Timeout configuration** | 30s timeout for LLM endpoints; 5s for CRUD endpoints | Prevents indefinite waiting; shows timeout error |

### Rendering Optimization

| Technique | Implementation | Where Applied |
|-----------|---------------|--------------|
| **Virtualized lists** | `@tanstack/react-virtual` for large lists | Application table with 100+ rows; chat history with many messages |
| **Memoization** | `React.memo()` for pure display components | `StatusBadge`, `ScoreCard`, `JobCard`, `ChatBubble` |
| **Stable references** | `useCallback` / `useMemo` for handlers and computed values | Event handlers passed to child components |
| **Selector optimization** | Zustand selectors to prevent re-renders | `useAuthStore((s) => s.apiKey)` instead of full store subscription |

### Scalability Considerations

| Aspect | Current Approach | Scaling Strategy |
|--------|-----------------|-----------------|
| **Multiple users** | Single `userId` in `authStore` | Add authentication (OAuth2/JWT); user ID derived from token |
| **Large conversation history** | Full conversation list in `chatStore` | Paginate conversations; load messages on demand; index by session |
| **Many applications** | Fetch all applications per user | Server-side pagination (`?page=1&limit=20`); implement infinite scroll |
| **Concurrent chat sessions** | One active session at a time | Multi-tab support via `BroadcastChannel` API for state sync |
| **Offline support** | No offline capability | Service Worker + IndexedDB for draft messages and cached application list |
| **Internationalization** | English only | Extract strings to `i18next` locale files; RTL support via Tailwind `rtl:` variant |
| **Real-time updates** | Polling for async tasks | Upgrade to SSE or WebSocket for server-pushed task status updates (backend change required) |

### Performance Budgets

| Metric | Target | Tool |
|--------|--------|------|
| **Largest Contentful Paint (LCP)** | < 2.5s | Lighthouse |
| **First Input Delay (FID)** | < 100ms | Lighthouse |
| **Cumulative Layout Shift (CLS)** | < 0.1 | Lighthouse |
| **Initial bundle size (gzipped)** | < 150 KB | Vite build stats |
| **Time to Interactive (TTI)** | < 3.5s | Lighthouse |
| **API response rendering** | < 200ms from response received to DOM update | Custom performance marks |

---

## Implementation Sequence

### Phase A: Foundation & Layout (Week 1–2)

| Step | Task | Depends On |
|------|------|-----------|
| A.1 | Scaffold Vite + React + TypeScript project | — |
| A.2 | Configure Tailwind CSS + shadcn/ui + ESLint + Prettier | A.1 |
| A.3 | Set up API client (Axios instance + interceptors) | A.1 |
| A.4 | Create TypeScript interfaces for all API types (`api/types.ts`) | A.3 |
| A.5 | Implement typed endpoint functions (`api/endpoints.ts`) | A.3, A.4 |
| A.6 | Create Zustand stores (`authStore`, `uiStore`) | A.1 |
| A.7 | Build shell layout (`AppShell`, `Sidebar`, `TopBar`) | A.2 |
| A.8 | Configure React Router with all routes + lazy loading | A.7 |
| A.9 | Implement `HealthIndicator` + Settings page | A.5, A.6 |
| A.10 | Set up TanStack Query provider + devtools | A.1 |
| A.11 | Build shared components (`LoadingSkeleton`, `ErrorBoundary`, `ErrorAlert`, `EmptyState`, `Toast`) | A.2 |
| A.12 | Implement Dashboard page (health + quick actions) | A.7, A.9, A.11 |

### Phase B: Core Features (Week 3–5)

| Step | Task | Depends On |
|------|------|-----------|
| B.1 | Build `ChatPage` + `ChatWindow` + `ChatBubble` + `ChatInput` + `TypingIndicator` | A.8, A.11 |
| B.2 | Create `chatStore` + `useChat` hook | A.6, A.5 |
| B.3 | Wire chat to `POST /chat` with attachment support | B.1, B.2 |
| B.4 | Build `ResumePage` + `ResumeUpload` + `ScoreCard` + analysis components | A.11 |
| B.5 | Create `useResumeAnalysis` hook + wire to `POST /resume` and `POST /resume/improve` | A.5 |
| B.6 | Add async resume audit with `useAsyncTask` + `AsyncTaskBanner` | B.5 |
| B.7 | Build `InterviewPage` + `SessionSetup` + `QuestionCard` + `AnswerInput` + `FeedbackPanel` | A.11 |
| B.8 | Create `interviewStore` + `useInterview` hook | A.6, A.5 |
| B.9 | Wire full interview flow (start → answer → complete → report) | B.7, B.8 |
| B.10 | Build `KnowledgePage` + answer display with Markdown + sources | A.11 |
| B.11 | Build `JobSearchPage` + `SearchForm` + `JobCard` list + `LocationMap` | A.11 |

### Phase C: Application Tracker (Week 6)

| Step | Task | Depends On |
|------|------|-----------|
| C.1 | Build `ApplicationTable` with sorting + filtering | A.11 |
| C.2 | Build `ApplicationForm` (create/edit modal) | C.1 |
| C.3 | Create `useApplications` hook with optimistic updates | A.5, A.10 |
| C.4 | Implement `StatusBadge` + `StatusWorkflow` diagram | C.1 |
| C.5 | Add `FollowUpAlert` banner + follow-ups integration | C.3 |
| C.6 | Add batch update with async task polling | C.3, B.6 |

### Phase D: User Profile & Memory (Week 7)

| Step | Task | Depends On |
|------|------|-----------|
| D.1 | Build `UserProfilePage` + `ProfileCard` + edit form | A.11 |
| D.2 | Create `useUserProfile` hook | A.5 |
| D.3 | Build `InsightsPanel` + `AgentUsageChart` (Recharts) | D.2 |
| D.4 | Add conversation history list to profile page | D.2 |
| D.5 | Integrate follow-up reminders into Dashboard | C.5 |
| D.6 | Add recent activity to Dashboard from user context | D.2, A.12 |

### Phase E: Polish & Testing (Week 8–9)

| Step | Task | Depends On |
|------|------|-----------|
| E.1 | Set up Vitest + React Testing Library | A.1 |
| E.2 | Write unit tests for all shared components | E.1, A.11 |
| E.3 | Write unit tests for all stores and hooks | E.1 |
| E.4 | Set up MSW handlers for all endpoints | E.1 |
| E.5 | Write integration tests for all pages | E.4 |
| E.6 | Set up Playwright + write E2E tests for critical paths | All B, C, D steps |
| E.7 | Accessibility audit (axe-core) on all pages | E.6 |
| E.8 | Add dark mode support | A.7 |
| E.9 | Responsive testing on mobile viewports (360px, 768px) | E.6 |
| E.10 | Error boundary testing (simulate component crashes) | A.11 |

### Phase F: Performance & Deployment (Week 10)

| Step | Task | Depends On |
|------|------|-----------|
| F.1 | Audit bundle size; add lazy imports for heavy libraries | E.6 |
| F.2 | Configure production Vite build (compression, sourcemaps) | F.1 |
| F.3 | Add Dockerfile for frontend (nginx static server) | F.2 |
| F.4 | Update `docker-compose.yml` to include frontend service | F.3 |
| F.5 | Configure CI/CD pipeline (lint → type-check → test → build → deploy) | E.6, F.2 |
| F.6 | Lighthouse performance audit; optimize to meet budgets | F.2 |
| F.7 | Write frontend-specific README with setup and development instructions | F.5 |
