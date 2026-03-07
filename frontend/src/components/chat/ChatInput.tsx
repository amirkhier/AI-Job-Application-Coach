import { useState, useRef, type FormEvent } from 'react';
import { Send, Paperclip, X, FileText, Loader2 } from 'lucide-react';
import { cn } from '@/lib/cn';
import { extractTextFromFile } from '@/lib/pdfExtract';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export function ChatInput({ onSend, disabled = false, placeholder = 'Type your message...' }: ChatInputProps) {
  const [value, setValue] = useState('');
  const [attachment, setAttachment] = useState<{ name: string; text: string } | null>(null);
  const [fileLoading, setFileLoading] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = value.trim();
    if ((!trimmed && !attachment) || disabled) return;

    let message = trimmed;
    if (attachment) {
      const fileContext = `[Attached file: ${attachment.name}]\n\n${attachment.text}`;
      message = message ? `${message}\n\n${fileContext}` : `Please analyze this file:\n\n${fileContext}`;
    }

    onSend(message);
    setValue('');
    setAttachment(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    // Reset input so re-selecting the same file triggers onChange
    e.target.value = '';

    setFileError(null);
    setFileLoading(true);
    try {
      const text = await extractTextFromFile(file);
      if (!text.trim()) {
        setFileError('Could not extract text from the file.');
        return;
      }
      setAttachment({ name: file.name, text: text.trim() });
    } catch {
      setFileError('Failed to read file. Please try a different format.');
    } finally {
      setFileLoading(false);
    }
  };

  return (
    <div className="space-y-2">
      {/* Attachment preview */}
      {attachment && (
        <div className="flex items-center gap-2 rounded-lg bg-primary/10 px-3 py-2 text-sm">
          <FileText size={14} className="shrink-0 text-primary" />
          <span className="truncate text-foreground">{attachment.name}</span>
          <span className="text-xs text-muted-foreground">
            ({Math.round(attachment.text.length / 1000)}k chars)
          </span>
          <button
            type="button"
            onClick={() => setAttachment(null)}
            className="ml-auto shrink-0 rounded p-0.5 text-muted-foreground hover:text-foreground transition-colors"
            aria-label="Remove attachment"
          >
            <X size={14} />
          </button>
        </div>
      )}

      {/* File error */}
      {fileError && (
        <p className="text-xs text-destructive px-1">{fileError}</p>
      )}

      {/* Input row */}
      <form onSubmit={handleSubmit} className="flex items-end gap-2">
        {/* File upload button */}
        <button
          type="button"
          disabled={disabled || fileLoading}
          onClick={() => fileInputRef.current?.click()}
          className={cn(
            'flex size-10 shrink-0 items-center justify-center rounded-xl border border-border bg-card text-muted-foreground transition-all hover:text-foreground hover:bg-muted/50',
            (disabled || fileLoading) && 'opacity-50 cursor-not-allowed',
          )}
          aria-label="Attach file"
          title="Attach a file (PDF, TXT, MD)"
        >
          {fileLoading ? <Loader2 size={16} className="animate-spin" /> : <Paperclip size={16} />}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.txt,.md,.doc,.docx"
          className="hidden"
          onChange={handleFileSelect}
        />

        <div className="relative flex-1">
          <textarea
            rows={1}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={attachment ? `Add a message about ${attachment.name}...` : placeholder}
            disabled={disabled}
            className={cn(
              'w-full resize-none rounded-xl border border-border bg-card px-4 py-3 pr-12 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-colors',
              disabled && 'opacity-50 cursor-not-allowed',
            )}
            style={{ maxHeight: '120px' }}
            onInput={(e) => {
              const target = e.target as HTMLTextAreaElement;
              target.style.height = 'auto';
              target.style.height = Math.min(target.scrollHeight, 120) + 'px';
            }}
          />
        </div>
        <button
          type="submit"
          disabled={disabled || (!value.trim() && !attachment)}
          className={cn(
            'flex size-10 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground transition-all hover:bg-primary/90',
            (disabled || (!value.trim() && !attachment)) && 'opacity-50 cursor-not-allowed',
          )}
          aria-label="Send message"
        >
          <Send size={16} />
        </button>
      </form>
    </div>
  );
}
