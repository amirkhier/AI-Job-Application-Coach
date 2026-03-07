import { useCallback } from 'react';
import { useMutation } from '@tanstack/react-query';
import { sendChatMessage } from '@/api/endpoints';
import { useChatStore, useAuthStore } from '@/stores';
import type { ChatRequest } from '@/api/types';

export function useChat() {
  const store = useChatStore();
  const userId = useAuthStore((s) => s.userId);

  const mutation = useMutation({
    mutationFn: (data: ChatRequest) => sendChatMessage(data),
    onSuccess: (response) => {
      const sessionId = store.activeSessionId!;
      store.addAssistantMessage(sessionId, response);
      store.setStreaming(false);
    },
    onError: () => {
      store.setStreaming(false);
    },
  });

  const sendMessage = useCallback(
    (content: string, sessionIdOrExtra?: string | Partial<ChatRequest>) => {
      let sessionId = store.activeSessionId;
      let extra: Partial<ChatRequest> = {};

      if (typeof sessionIdOrExtra === 'string') {
        sessionId = sessionIdOrExtra;
        store.setActiveSession(sessionId);
      } else if (sessionIdOrExtra) {
        extra = sessionIdOrExtra;
      }

      if (!sessionId) {
        sessionId = store.createSession();
      }
      store.addUserMessage(sessionId, content);
      store.setStreaming(true);

      mutation.mutate({
        message: content,
        user_id: userId,
        session_id: sessionId,
        ...extra,
      });
    },
    [store, mutation, userId],
  );

  const currentMessages = store.activeSessionId
    ? store.sessions[store.activeSessionId] ?? []
    : [];

  return {
    sendMessage,
    messages: currentMessages,
    isLoading: mutation.isPending,
    isStreaming: store.isStreaming,
    activeSessionId: store.activeSessionId,
    sessions: store.sessions,
    createSession: store.createSession,
    setActiveSession: store.setActiveSession,
    deleteSession: store.deleteSession,
    clearSession: store.clearSession,
    error: mutation.error,
    reset: mutation.reset,
  };
}
