import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Bot, User, Clock, Zap } from 'lucide-react';
import { cn } from '@/lib/cn';
import type { ChatMessage } from '@/stores';
import { formatConfidence, formatDuration } from '@/lib/formatters';

interface ChatBubbleProps {
  message: ChatMessage;
}

export function ChatBubble({ message }: ChatBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div className={cn('flex gap-3', isUser ? 'flex-row-reverse' : 'flex-row')}>
      <div
        className={cn(
          'flex size-8 shrink-0 items-center justify-center rounded-full',
          isUser ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground',
        )}
      >
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>
      <div
        className={cn(
          'max-w-[75%] space-y-2 rounded-2xl px-4 py-3',
          isUser
            ? 'bg-primary text-primary-foreground rounded-br-md'
            : 'bg-card border border-border text-foreground rounded-bl-md',
        )}
      >
        <div className={cn('prose prose-sm max-w-none', isUser ? 'prose-invert' : 'dark:prose-invert')}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
        </div>
        {!isUser && (message.intent || message.processingTime) && (
          <div className="flex items-center gap-3 pt-1 border-t border-border/30">
            {message.intent && (
              <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                <Zap size={10} />
                {message.intent}
                {message.confidence != null && ` (${formatConfidence(message.confidence)})`}
              </span>
            )}
            {message.processingTime != null && (
              <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                <Clock size={10} />
                {formatDuration(message.processingTime)}
              </span>
            )}
          </div>
        )}
        {!isUser && message.agentsUsed && message.agentsUsed.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {message.agentsUsed.map((agent) => (
              <span key={agent} className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
                {agent}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
