import { useEffect, useRef } from 'react';
import { Plus, Trash2, MessageSquare } from 'lucide-react';
import { cn } from '@/lib/cn';
import { useChatStore } from '@/stores';
import { useChat } from '@/hooks';
import { ChatBubble, ChatInput, TypingIndicator } from '@/components/chat';
import { EmptyState } from '@/components/shared/EmptyState';
import { ErrorAlert } from '@/components/shared/ErrorAlert';

export default function ChatPage() {
  const {
    sessions,
    activeSessionId,
    isStreaming,
    setActiveSession,
    createSession,
    deleteSession,
  } = useChatStore();

  const { sendMessage, error, reset } = useChat();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const activeMessages = activeSessionId ? sessions[activeSessionId] ?? [] : [];
  const sessionIds = Object.keys(sessions);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeMessages.length, isStreaming]);

  const handleSend = (message: string) => {
    if (!activeSessionId) {
      const id = createSession();
      sendMessage(message, id);
    } else {
      sendMessage(message, activeSessionId);
    }
  };

  const handleNewSession = () => {
    createSession();
    reset();
  };

  return (
    <div className="flex h-[calc(100vh-7rem)] gap-4">
      {/* Sessions sidebar */}
      <div className="hidden md:flex w-56 shrink-0 flex-col rounded-xl border border-border bg-card">
        <div className="flex items-center justify-between border-b border-border px-3 py-2">
          <span className="text-sm font-semibold text-foreground">Sessions</span>
          <button
            onClick={handleNewSession}
            className="rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            aria-label="New session"
          >
            <Plus size={16} />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {sessionIds.length === 0 && (
            <p className="px-2 py-4 text-xs text-center text-muted-foreground">No sessions yet</p>
          )}
          {sessionIds.map((id) => {
            const msgs = sessions[id] ?? [];
            const preview = msgs[0]?.content ?? 'New session';
            return (
              <button
                key={id}
                onClick={() => setActiveSession(id)}
                className={cn(
                  'w-full flex items-center gap-2 rounded-lg px-2.5 py-2 text-left text-sm transition-colors group',
                  id === activeSessionId
                    ? 'bg-primary/10 text-primary'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground',
                )}
              >
                <MessageSquare size={14} className="shrink-0" />
                <span className="flex-1 truncate">{preview.slice(0, 30)}</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteSession(id);
                  }}
                  className="shrink-0 opacity-0 group-hover:opacity-100 rounded p-0.5 text-muted-foreground hover:text-destructive transition-all"
                  aria-label="Delete session"
                >
                  <Trash2 size={12} />
                </button>
              </button>
            );
          })}
        </div>
      </div>

      {/* Chat area */}
      <div className="flex flex-1 flex-col rounded-xl border border-border bg-card overflow-hidden">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {activeMessages.length === 0 && !error && (
            <EmptyState
              icon={<MessageSquare size={40} strokeWidth={1.5} />}
              title="Start a Conversation"
              description="Ask me anything about your job search — resume help, interview tips, career advice, or job searching."
              className="h-full border-none bg-transparent"
            />
          )}
          {error && <ErrorAlert error={error} onRetry={reset} onDismiss={reset} />}
          {activeMessages.map((msg) => (
            <ChatBubble key={msg.id} message={msg} />
          ))}
          {isStreaming && <TypingIndicator />}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t border-border p-4">
          <ChatInput
            onSend={handleSend}
            disabled={isStreaming}
            placeholder={activeSessionId ? 'Type your message...' : 'Start a new conversation...'}
          />
        </div>
      </div>
    </div>
  );
}
