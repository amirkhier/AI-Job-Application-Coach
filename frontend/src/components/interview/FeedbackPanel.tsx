import { CheckCircle, XCircle, MessageSquare, Lightbulb } from 'lucide-react';
import type { AnswerFeedback } from '@/api/types';
import { ScoreCard } from '@/components/resume/ScoreCard';

interface FeedbackPanelProps {
  feedback: AnswerFeedback;
}

export function FeedbackPanel({ feedback }: FeedbackPanelProps) {
  return (
    <div className="rounded-xl border border-border bg-card p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-foreground">Answer Feedback</h3>
        <ScoreCard label="Score" score={feedback.overall_score} className="w-24 p-2" />
      </div>

      {(feedback?.strength_areas ?? []).length > 0 && (
        <div>
          <p className="flex items-center gap-1.5 text-xs font-medium text-success mb-1.5">
            <CheckCircle size={12} /> Strengths
          </p>
          <ul className="space-y-1">
            {(feedback.strength_areas ?? []).map((s, i) => (
              <li key={i} className="text-sm text-foreground">{s}</li>
            ))}
          </ul>
        </div>
      )}

      {(feedback?.improvement_areas ?? []).length > 0 && (
        <div>
          <p className="flex items-center gap-1.5 text-xs font-medium text-warning mb-1.5">
            <XCircle size={12} /> Areas to Improve
          </p>
          <ul className="space-y-1">
            {(feedback.improvement_areas ?? []).map((a, i) => (
              <li key={i} className="text-sm text-foreground">{a}</li>
            ))}
          </ul>
        </div>
      )}

      {feedback.specific_feedback && (
        <div>
          <p className="flex items-center gap-1.5 text-xs font-medium text-primary mb-1.5">
            <MessageSquare size={12} /> Feedback
          </p>
          <p className="text-sm text-muted-foreground">{feedback.specific_feedback}</p>
        </div>
      )}

      {feedback.suggested_improvement && (
        <div>
          <p className="flex items-center gap-1.5 text-xs font-medium text-accent mb-1.5">
            <Lightbulb size={12} /> Suggested Improvement
          </p>
          <p className="text-sm text-muted-foreground">{feedback.suggested_improvement}</p>
        </div>
      )}
    </div>
  );
}
