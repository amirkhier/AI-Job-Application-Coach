import { create } from 'zustand';
import type { InterviewQuestion, AnswerFeedback, SessionSummary } from '@/api/types';

export interface AnswerRecord {
  questionId: string;
  answer: string;
  feedback: AnswerFeedback;
}

interface InterviewState {
  sessionId: string | null;
  role: string;
  level: string;
  currentQuestion: InterviewQuestion | null;
  answers: AnswerRecord[];
  totalQuestions: number;
  isComplete: boolean;
  summary: SessionSummary | null;
  isSubmitting: boolean;

  startSession: (
    sessionId: string,
    role: string,
    level: string,
    firstQuestion: InterviewQuestion,
    totalQuestions: number,
  ) => void;
  recordAnswer: (record: AnswerRecord) => void;
  setCurrentQuestion: (question: InterviewQuestion | null) => void;
  completeSession: (summary: SessionSummary) => void;
  setSubmitting: (submitting: boolean) => void;
  resetSession: () => void;
}

const initialState = {
  sessionId: null,
  role: '',
  level: '',
  currentQuestion: null,
  answers: [],
  totalQuestions: 0,
  isComplete: false,
  summary: null,
  isSubmitting: false,
};

export const useInterviewStore = create<InterviewState>()((set) => ({
  ...initialState,

  startSession: (sessionId, role, level, firstQuestion, totalQuestions) =>
    set({
      ...initialState,
      sessionId,
      role,
      level,
      currentQuestion: firstQuestion,
      totalQuestions,
    }),

  recordAnswer: (record) =>
    set((s) => ({ answers: [...s.answers, record] })),

  setCurrentQuestion: (question) => set({ currentQuestion: question }),

  completeSession: (summary) =>
    set({ isComplete: true, summary, currentQuestion: null }),

  setSubmitting: (submitting) => set({ isSubmitting: submitting }),

  resetSession: () => set(initialState),
}));
