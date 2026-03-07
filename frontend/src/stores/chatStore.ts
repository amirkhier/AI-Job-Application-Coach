import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ChatResponse } from '@/api/types';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  intent?: string;
  confidence?: number;
  agentsUsed?: string[];
  processingTime?: number;
  data?: Record<string, unknown> | null;
  interviewSessionId?: string | null;
}

interface ChatState {
  sessions: Record<string, ChatMessage[]>;
  activeSessionId: string | null;
  isStreaming: boolean;

  setActiveSession: (sessionId: string) => void;
  createSession: () => string;
  addUserMessage: (sessionId: string, content: string) => void;
  addAssistantMessage: (sessionId: string, response: ChatResponse) => void;
  clearSession: (sessionId: string) => void;
  deleteSession: (sessionId: string) => void;
  setStreaming: (streaming: boolean) => void;
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      sessions: {},
      activeSessionId: null,
      isStreaming: false,

      setActiveSession: (sessionId) => set({ activeSessionId: sessionId }),

      createSession: () => {
        const id = crypto.randomUUID();
        set((s) => ({
          sessions: { ...s.sessions, [id]: [] },
          activeSessionId: id,
        }));
        return id;
      },

      addUserMessage: (sessionId, content) => {
        const msg: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'user',
          content,
          timestamp: new Date().toISOString(),
        };
        set((s) => ({
          sessions: {
            ...s.sessions,
            [sessionId]: [...(s.sessions[sessionId] ?? []), msg],
          },
        }));
      },

      addAssistantMessage: (sessionId, response) => {
        const msg: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: response.response,
          timestamp: new Date().toISOString(),
          intent: response.intent,
          confidence: response.confidence,
          agentsUsed: response.agents_used,
          processingTime: response.processing_time,
          data: response.data,
          interviewSessionId: response.interview_session_id,
        };
        set((s) => ({
          sessions: {
            ...s.sessions,
            [sessionId]: [...(s.sessions[sessionId] ?? []), msg],
          },
        }));
      },

      clearSession: (sessionId) =>
        set((s) => ({
          sessions: { ...s.sessions, [sessionId]: [] },
        })),

      deleteSession: (sessionId) => {
        const { sessions, activeSessionId } = get();
        const { [sessionId]: _, ...rest } = sessions;
        set({
          sessions: rest,
          activeSessionId: activeSessionId === sessionId ? null : activeSessionId,
        });
      },

      setStreaming: (streaming) => set({ isStreaming: streaming }),
    }),
    {
      name: 'chat-sessions',
      partialize: (state) => ({
        sessions: state.sessions,
        activeSessionId: state.activeSessionId,
      }),
    },
  ),
);
