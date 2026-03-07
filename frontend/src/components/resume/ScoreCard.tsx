import { cn } from '@/lib/cn';
import { formatScore } from '@/lib/formatters';

interface ScoreCardProps {
  label: string;
  score: number;
  max?: number;
  className?: string;
}

export function ScoreCard({ label, score, max = 10, className }: ScoreCardProps) {
  const percentage = (score / max) * 100;
  const color =
    percentage >= 80
      ? 'text-success'
      : percentage >= 60
        ? 'text-warning'
        : 'text-destructive';

  const bgColor =
    percentage >= 80
      ? 'bg-success'
      : percentage >= 60
        ? 'bg-warning'
        : 'bg-destructive';

  return (
    <div className={cn('rounded-xl border border-border bg-card p-4 text-center', className)}>
      <p className="text-xs text-muted-foreground mb-2">{label}</p>
      <p className={cn('text-3xl font-bold', color)}>{formatScore(score)}</p>
      <p className="text-xs text-muted-foreground mt-1">out of {max}</p>
      <div className="mt-3 h-2 rounded-full bg-muted overflow-hidden">
        <div
          className={cn('h-full rounded-full transition-all duration-500', bgColor)}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
    </div>
  );
}
