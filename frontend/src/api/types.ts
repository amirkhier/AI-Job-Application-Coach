// ─── Health ──────────────────────────────────────────────
export interface HealthResponse {
  status: 'healthy' | 'degraded';
  service: string;
  timestamp: string;
  database_connected: boolean;
  checks: {
    database: boolean;
    redis: boolean;
    chromadb: boolean;
  };
}

// ─── Resume ──────────────────────────────────────────────
export interface ResumeRequest {
  resume_text: string;
  job_description?: string;
  user_id?: number;
}

export interface ResumeResponse {
  overall_score: number;
  strengths: string[];
  weaknesses: string[];
  recommendations: string[];
  ats_compatibility: {
    score: number;
    issues: string[];
    suggestions: string[];
  };
  keyword_analysis: {
    present_keywords: string[];
    missing_keywords: string[];
    keyword_density_notes: string;
  };
  section_feedback: Record<string, string>;
  processing_time: number;
}

export interface ResumeImprovementResponse {
  improved_summary: string;
  improved_bullets: {
    original: string;
    improved: string;
    reasoning: string;
  }[];
  additional_suggestions: string[];
  priority_actions: string[];
  processing_time: number;
}

// ─── Interview ───────────────────────────────────────────
export interface InterviewStartRequest {
  role: string;
  level?: string;
  question_count?: number;
  user_id?: number;
}

export interface InterviewQuestion {
  id: string;
  question: string;
  type: string;
  difficulty: string;
  key_points: string[];
}

export interface InterviewStartResponse {
  session_id: string;
  role: string;
  level: string;
  first_question: InterviewQuestion;
  total_questions: number;
}

export interface InterviewAnswerRequest {
  session_id: string;
  question_id: string;
  answer: string;
}

export interface AnswerFeedback {
  overall_score: number;
  strength_areas: string[];
  improvement_areas: string[];
  specific_feedback: string;
  suggested_improvement: string;
}

export interface InterviewAnswerResponse {
  feedback: AnswerFeedback;
  next_question: InterviewQuestion | null;
  session_complete: boolean;
  session_summary: SessionSummary | null;
}

export interface SessionSummary {
  overall_score: number;
  total_questions: number;
  strengths?: string[];
  weaknesses?: string[];
  recommendations?: string[];
  // Backend may return these alternate field names
  strongest_areas?: string[];
  weakest_areas?: string[];
  key_recommendations?: string[];
  [key: string]: unknown;
}

// ─── Knowledge ───────────────────────────────────────────
export interface KnowledgeQueryRequest {
  query: string;
  user_id?: number;
}

export interface KnowledgeQueryResponse {
  answer: string;
  sources: string[];
  relevance_score: number;
  related_topics: string[];
}

// ─── Jobs ────────────────────────────────────────────────
export interface JobSearchRequest {
  query: string;
  location?: string;
  experience_level?: string;
  remote_ok?: boolean;
  count?: number;
  user_id?: number;
}

export interface JobListing {
  title: string;
  company: string;
  location: string;
  description: string;
  url: string | null;
  salary_range: string;
  remote_friendly: boolean;
  match_score: number | null;
  experience_level: string;
  key_skills: string[];
}

export interface LocationInfo {
  city: string;
  lat: number;
  lon: number;
  display_name: string;
  country: string;
  found: boolean;
}

export interface NearbyCompany {
  name: string;
  type: string;
  distance_m: number;
  address: string;
}

export interface JobSearchResponse {
  jobs: JobListing[];
  total_found: number;
  search_query: string;
  location: string;
  location_info: LocationInfo | null;
  nearby_companies: NearbyCompany[];
  processing_time: number;
}

// ─── Applications ────────────────────────────────────────
export type ApplicationStatus = 'applied' | 'interviewing' | 'offer' | 'rejected' | 'withdrawn';

export interface ApplicationCreateRequest {
  company_name: string;
  position_title: string;
  job_url?: string;
  application_date?: string;
  notes?: string;
  user_id?: number;
}

export interface ApplicationUpdateRequest {
  status?: ApplicationStatus;
  notes?: string;
  follow_up_date?: string;
}

export interface ApplicationResponse {
  id: number;
  company_name: string;
  position_title: string;
  job_url: string | null;
  status: ApplicationStatus;
  application_date: string;
  follow_up_date: string | null;
  notes: string | null;
  created_at: string;
  updated_at: string;
}

// ─── User / Memory ───────────────────────────────────────
export interface UserProfile {
  id: number;
  skills: string[];
  experience_level: string;
  experience_years: number;
  target_roles: string[];
  career_goals: string;
  preferences: Record<string, unknown>;
}

export interface ConversationRecord {
  intent: string;
  summary: string;
  timestamp: string;
  agents_used: string[];
}

export interface UserContextResponse {
  user_profile: UserProfile;
  recent_conversations: ConversationRecord[];
  context_summary: {
    total_conversations: number;
    top_intents: string[];
  };
  history_count: number;
}

export interface ConversationAnalysisResponse {
  insights: string[];
  patterns: string[];
  recommendations: string[];
  conversation_count: number;
  agent_usage: Record<string, number>;
}

export interface ProfileUpdateRequest {
  skills?: string[];
  experience_level?: string;
  experience_years?: number;
  target_roles?: string[];
  career_goals?: string;
}

// ─── Chat ────────────────────────────────────────────────
export interface ChatRequest {
  message: string;
  user_id?: number;
  session_id?: string;
  resume_text?: string;
  job_description?: string;
  interview_role?: string;
  interview_level?: string;
  job_search_location?: string;
  interview_session_id?: string;
  interview_answer?: string;
  interview_question_id?: string;
}

export interface ChatResponse {
  response: string;
  intent: string;
  confidence: number;
  agents_used: string[];
  session_id: string;
  processing_time: number;
  error: string | null;
  data: Record<string, unknown> | null;
  interview_session_id: string | null;
}

// ─── Async Tasks ─────────────────────────────────────────
export interface AsyncTaskResponse {
  task_id: string;
  status: string;
  message: string;
  estimated_completion?: string;
}

export interface TaskStatusResponse {
  task_id: string;
  status: 'PENDING' | 'STARTED' | 'SUCCESS' | 'FAILURE';
  result?: unknown;
}

// ─── App Error ───────────────────────────────────────────
export interface AppError {
  code: number;
  message: string;
  detail?: string;
  requestId?: string;
  retryable: boolean;
}
