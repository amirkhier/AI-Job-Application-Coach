import { ArrowRight, Lightbulb, ListChecks } from 'lucide-react';
import type { ResumeImprovementResponse } from '@/api/types';

interface ImprovementResultsProps {
  data: ResumeImprovementResponse;
}

export function ImprovementResults({ data }: ImprovementResultsProps) {
  return (
    <div className="space-y-6">
      {/* Improved Summary */}
      {data.improved_summary && (
        <section>
          <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground mb-3">
            <Lightbulb size={16} className="text-warning" />
            Improved Summary
          </h3>
          <div className="rounded-lg border border-border bg-background p-4">
            <p className="text-sm text-foreground whitespace-pre-wrap">{data.improved_summary}</p>
          </div>
        </section>
      )}

      {/* Improved Bullets */}
      {data.improved_bullets.length > 0 && (
        <section>
          <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground mb-3">
            <ArrowRight size={16} className="text-primary" />
            Improved Bullet Points
          </h3>
          <div className="space-y-3">
            {data.improved_bullets.map((bullet, i) => (
              <div key={i} className="rounded-lg border border-border bg-background p-4 space-y-2">
                <div className="flex items-start gap-2">
                  <span className="shrink-0 rounded bg-destructive/10 px-1.5 py-0.5 text-[10px] font-medium text-destructive">Before</span>
                  <p className="text-sm text-muted-foreground line-through">{bullet.original}</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="shrink-0 rounded bg-success/10 px-1.5 py-0.5 text-[10px] font-medium text-success">After</span>
                  <p className="text-sm text-foreground">{bullet.improved}</p>
                </div>
                <p className="text-xs text-muted-foreground italic">{bullet.reasoning}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Priority Actions */}
      {data.priority_actions.length > 0 && (
        <section>
          <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground mb-3">
            <ListChecks size={16} className="text-success" />
            Priority Actions
          </h3>
          <ol className="space-y-1.5 list-decimal list-inside">
            {data.priority_actions.map((action, i) => (
              <li key={i} className="text-sm text-foreground">{action}</li>
            ))}
          </ol>
        </section>
      )}

      {/* Additional Suggestions */}
      {data.additional_suggestions.length > 0 && (
        <section>
          <h3 className="text-sm font-semibold text-foreground mb-3">Additional Suggestions</h3>
          <ul className="space-y-1.5">
            {data.additional_suggestions.map((s, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-foreground">
                <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-primary" />
                {s}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
