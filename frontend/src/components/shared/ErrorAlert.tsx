import { AlertCircle, XCircle, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/cn';
import type { AppError } from '@/api/types';

interface ErrorAlertProps {
  error: AppError | Error | unknown;
  onRetry?: () => void;
  onDismiss?: () => void;
  className?: string;
}

function extractMessage(error: unknown): string {
  if (!error) return 'Unknown error';
  if (typeof error === 'object' && error !== null) {
    if ('message' in error) return String((error as { message: string }).message);
  }
  return String(error);
}

function isRetryable(error: unknown): boolean {
  if (typeof error === 'object' && error !== null && 'retryable' in error) {
    return Boolean((error as { retryable: boolean }).retryable);
  }
  return false;
}

export function ErrorAlert({ error, onRetry, onDismiss, className }: ErrorAlertProps) {
  const message = extractMessage(error);

  return (
    <div
      className={cn(
        'flex items-start gap-3 rounded-lg border border-destructive/30 bg-destructive/5 p-4',
        className,
      )}
      role="alert"
    >
      <AlertCircle className="mt-0.5 size-5 shrink-0 text-destructive" />
      <div className="flex-1 space-y-1">
        <p className="text-sm font-medium text-foreground">Error</p>
        <p className="text-sm text-muted-foreground">{message}</p>
      </div>
      <div className="flex items-center gap-1">
        {onRetry && isRetryable(error) && (
          <button
            onClick={onRetry}
            className="rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            aria-label="Retry"
          >
            <RefreshCw size={14} />
          </button>
        )}
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            aria-label="Dismiss"
          >
            <XCircle size={14} />
          </button>
        )}
      </div>
    </div>
  );
}
