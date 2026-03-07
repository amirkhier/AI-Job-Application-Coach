import { Bot } from 'lucide-react';

export function TypingIndicator() {
  return (
    <div className="flex gap-3">
      <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground">
        <Bot size={16} />
      </div>
      <div className="rounded-2xl rounded-bl-md bg-card border border-border px-4 py-3">
        <div className="flex items-center gap-1.5">
          <div className="size-2 animate-bounce rounded-full bg-muted-foreground/60" style={{ animationDelay: '0ms' }} />
          <div className="size-2 animate-bounce rounded-full bg-muted-foreground/60" style={{ animationDelay: '150ms' }} />
          <div className="size-2 animate-bounce rounded-full bg-muted-foreground/60" style={{ animationDelay: '300ms' }} />
        </div>
      </div>
    </div>
  );
}
