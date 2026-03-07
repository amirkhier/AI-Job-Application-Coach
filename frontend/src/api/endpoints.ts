import { apiClient } from './client';
import type {
  HealthResponse,
  ResumeRequest,
  ResumeResponse,
  ResumeImprovementResponse,
  InterviewStartRequest,
  InterviewStartResponse,
  InterviewAnswerRequest,
  InterviewAnswerResponse,
  InterviewQuestion,
  KnowledgeQueryRequest,
  KnowledgeQueryResponse,
  JobSearchRequest,
  JobSearchResponse,
  ApplicationCreateRequest,
  ApplicationResponse,
  ApplicationUpdateRequest,
  UserContextResponse,
  ConversationAnalysisResponse,
  ProfileUpdateRequest,
  UserProfile,
  ChatRequest,
  ChatResponse,
  AsyncTaskResponse,
  TaskStatusResponse,
} from './types';

// ─── Health ──────────────────────────────────────────────
export const getHealth = () =>
  apiClient.get<HealthResponse>('/health').then((r) => r.data);

// ─── Resume ──────────────────────────────────────────────
export const analyzeResume = (data: ResumeRequest) =>
  apiClient.post<ResumeResponse>('/resume', data).then((r) => r.data);

export const improveResume = (data: ResumeRequest) =>
  apiClient.post<ResumeImprovementResponse>('/resume/improve', data).then((r) => r.data);

export const auditResume = (data: ResumeRequest) =>
  apiClient.post<AsyncTaskResponse>('/resume/audit', data).then((r) => r.data);

// ─── Interview ───────────────────────────────────────────
export const startInterview = (data: InterviewStartRequest) =>
  apiClient.post<InterviewStartResponse>('/interview/start', data).then((r) => r.data);

export const submitAnswer = (data: InterviewAnswerRequest) =>
  apiClient.post<InterviewAnswerResponse>('/interview/answer', data).then((r) => r.data);

export const getQuestions = (sessionId: string) =>
  apiClient.get<InterviewQuestion[]>(`/interview/questions/${sessionId}`).then((r) => r.data);

export const generateReport = (sessionId: string) =>
  apiClient.post<AsyncTaskResponse>('/interview/report', { session_id: sessionId }).then((r) => r.data);

// ─── Knowledge ───────────────────────────────────────────
export const askQuestion = (data: KnowledgeQueryRequest) =>
  apiClient.post<KnowledgeQueryResponse>('/ask', data).then((r) => r.data);

// ─── Jobs ────────────────────────────────────────────────
export const searchJobs = (data: JobSearchRequest) =>
  apiClient.post<JobSearchResponse>('/jobs/search', data).then((r) => r.data);

export const matchJobs = (data: { resume_text: string; query?: string; location?: string }) =>
  apiClient.post<JobSearchResponse>('/jobs/match', data).then((r) => r.data);

export const getLocationInfo = (location: string) =>
  apiClient.get<{ location_info: unknown; nearby_companies: unknown[] }>(`/jobs/location/${encodeURIComponent(location)}`).then((r) => r.data);

// ─── Applications ────────────────────────────────────────
export const createApplication = (data: ApplicationCreateRequest) =>
  apiClient.post<ApplicationResponse>('/applications', data).then((r) => r.data);

export const listApplications = (params?: { user_id?: number; status?: string }) =>
  apiClient.get<ApplicationResponse[]>('/applications', { params }).then((r) => r.data);

export const getApplication = (id: number) =>
  apiClient.get<ApplicationResponse>(`/applications/${id}`).then((r) => r.data);

export const updateApplication = (id: number, data: ApplicationUpdateRequest) =>
  apiClient.put<ApplicationResponse>(`/applications/${id}`, data).then((r) => r.data);

export const deleteApplication = (id: number) =>
  apiClient.delete(`/applications/${id}`).then((r) => r.data);

export const getFollowUps = (userId?: number) =>
  apiClient.get<ApplicationResponse[]>('/applications/follow-ups', { params: { user_id: userId } }).then((r) => r.data);

export const batchUpdateApplications = (updates: { ids: number[]; status: string }) =>
  apiClient.post<{ updated: number }>('/applications/batch-update', updates).then((r) => r.data);

// ─── User / Memory ───────────────────────────────────────
export const getUserProfile = (userId: number) =>
  apiClient.get<UserProfile>(`/user/${userId}/profile`).then((r) => r.data);

export const getUserContext = (userId: number) =>
  apiClient.get<UserContextResponse>(`/user/${userId}/context`).then((r) => r.data);

export const getUserInsights = (userId: number) =>
  apiClient.get<ConversationAnalysisResponse>(`/user/${userId}/insights`).then((r) => r.data);

export const updateUserProfile = (userId: number, data: ProfileUpdateRequest) =>
  apiClient.put(`/user/${userId}/profile`, data).then((r) => r.data);

// ─── Chat ────────────────────────────────────────────────
export const sendChatMessage = (data: ChatRequest) =>
  apiClient.post<ChatResponse>('/chat', data).then((r) => r.data);

// ─── Async Tasks ─────────────────────────────────────────
export const getTaskStatus = (taskId: string) =>
  apiClient.get<TaskStatusResponse>(`/tasks/${taskId}/status`).then((r) => r.data);

export const getTaskResult = (taskId: string) =>
  apiClient.get<TaskStatusResponse>(`/result/${taskId}`).then((r) => r.data);
