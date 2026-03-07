import { HelpCircle, Tag, BarChart3 } from 'lucide-react';
import { cn } from '@/lib/cn';
import type { InterviewQuestion } from '@/api/types';

interface QuestionCardProps {
  question: InterviewQuestion;
  current: number;
  total: number;
}

export function QuestionCard({ question, current, total }: QuestionCardProps) {
  return (
    <div className="rounded-xl border border-border bg-card p-6 space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground">
          Question {current} of {total}
        </span>
        <div className="flex items-center gap-2">
          <span className={cn(
            'rounded-full px-2.5 py-0.5 text-xs font-medium',
            question.difficulty === 'hard'
              ? 'bg-destructive/10 text-destructive'
              : question.difficulty === 'medium'
                ? 'bg-warning/10 text-warning'
                : 'bg-success/10 text-success',
          )}>
            {question.difficulty}
          </span>
          <span className="rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary capitalize">
            {question.type}
          </span>
        </div>
      </div>

      <div className="flex items-start gap-3">
        <HelpCircle size={20} className="mt-0.5 shrink-0 text-primary" />
        <p className="text-base font-medium text-foreground">{question.question}</p>
      </div>

      {(question.key_points ?? []).length > 0 && (
        <div>
          <p className="text-xs font-medium text-muted-foreground mb-1.5">Key Points to Cover:</p>
          <div className="flex flex-wrap gap-1.5">
            {(question.key_points ?? []).map((point) => (
              <span key={point} className="rounded-full bg-muted px-2.5 py-0.5 text-xs text-muted-foreground">
                {point}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Progress bar */}
      <div className="h-1.5 rounded-full bg-muted overflow-hidden">
        <div
          className="h-full rounded-full bg-primary transition-all duration-500"
          style={{ width: `${(current / total) * 100}%` }}
        />
      </div>
    </div>
  );
}
