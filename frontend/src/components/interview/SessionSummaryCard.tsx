import { Trophy, CheckCircle, XCircle, Lightbulb } from 'lucide-react';
import type { SessionSummary } from '@/api/types';
import { ScoreCard } from '@/components/resume/ScoreCard';

interface SessionSummaryCardProps {
  summary: SessionSummary;
  role: string;
  level: string;
  onReset: () => void;
}

export function SessionSummaryCard({ summary, role, level, onReset }: SessionSummaryCardProps) {
  return (
    <div className="space-y-6">
      <div className="text-center space-y-2">
        <Trophy size={48} className="mx-auto text-warning" />
        <h2 className="text-xl font-bold text-foreground">Interview Complete!</h2>
        <p className="text-sm text-muted-foreground">
          {role} ({level}) — {summary.total_questions} questions
        </p>
      </div>

      <div className="flex justify-center">
        <ScoreCard label="Overall Score" score={summary.overall_score} className="w-40" />
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {(summary.strengths ?? summary.strongest_areas ?? []).length > 0 && (
          <div className="rounded-xl border border-border bg-card p-4">
            <h3 className="flex items-center gap-1.5 text-sm font-semibold text-success mb-2">
              <CheckCircle size={14} /> Strengths
            </h3>
            <ul className="space-y-1">
              {(summary.strengths ?? summary.strongest_areas ?? []).map((s, i) => (
                <li key={i} className="text-sm text-foreground">{s}</li>
              ))}
            </ul>
          </div>
        )}

        {(summary.weaknesses ?? summary.weakest_areas ?? []).length > 0 && (
          <div className="rounded-xl border border-border bg-card p-4">
            <h3 className="flex items-center gap-1.5 text-sm font-semibold text-destructive mb-2">
              <XCircle size={14} /> Areas to Improve
            </h3>
            <ul className="space-y-1">
              {(summary.weaknesses ?? summary.weakest_areas ?? []).map((w, i) => (
                <li key={i} className="text-sm text-foreground">{w}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {(summary.recommendations ?? summary.key_recommendations ?? []).length > 0 && (
        <div className="rounded-xl border border-border bg-card p-4">
          <h3 className="flex items-center gap-1.5 text-sm font-semibold text-warning mb-2">
            <Lightbulb size={14} /> Recommendations
          </h3>
          <ul className="space-y-1">
            {(summary.recommendations ?? summary.key_recommendations ?? []).map((r, i) => (
              <li key={i} className="text-sm text-foreground">{r}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="flex justify-center">
        <button
          onClick={onReset}
          className="rounded-xl bg-primary px-8 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          Start New Interview
        </button>
      </div>
    </div>
  );
}
