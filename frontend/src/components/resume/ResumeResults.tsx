import { CheckCircle, XCircle, AlertTriangle, Tag, FileSearch } from 'lucide-react';
import { cn } from '@/lib/cn';
import type { ResumeResponse } from '@/api/types';

interface ResumeResultsProps {
  data: ResumeResponse;
}

export function ResumeResults({ data }: ResumeResultsProps) {
  return (
    <div className="space-y-6">
      {/* Strengths */}
      {data.strengths.length > 0 && (
        <section>
          <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground mb-3">
            <CheckCircle size={16} className="text-success" />
            Strengths
          </h3>
          <ul className="space-y-1.5">
            {data.strengths.map((s, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-foreground">
                <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-success" />
                {s}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* Weaknesses */}
      {data.weaknesses.length > 0 && (
        <section>
          <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground mb-3">
            <XCircle size={16} className="text-destructive" />
            Weaknesses
          </h3>
          <ul className="space-y-1.5">
            {data.weaknesses.map((w, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-foreground">
                <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-destructive" />
                {w}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* Recommendations */}
      {data.recommendations.length > 0 && (
        <section>
          <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground mb-3">
            <AlertTriangle size={16} className="text-warning" />
            Recommendations
          </h3>
          <ul className="space-y-1.5">
            {data.recommendations.map((r, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-foreground">
                <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-warning" />
                {r}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* ATS Compatibility */}
      <section>
        <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground mb-3">
          <FileSearch size={16} className="text-primary" />
          ATS Compatibility — {data.ats_compatibility.score}/10
        </h3>
        {data.ats_compatibility.issues.length > 0 && (
          <div className="mb-3">
            <p className="text-xs font-medium text-muted-foreground mb-1.5">Issues</p>
            <ul className="space-y-1">
              {data.ats_compatibility.issues.map((issue, i) => (
                <li key={i} className="text-sm text-destructive">{issue}</li>
              ))}
            </ul>
          </div>
        )}
        {data.ats_compatibility.suggestions.length > 0 && (
          <div>
            <p className="text-xs font-medium text-muted-foreground mb-1.5">Suggestions</p>
            <ul className="space-y-1">
              {data.ats_compatibility.suggestions.map((s, i) => (
                <li key={i} className="text-sm text-foreground">{s}</li>
              ))}
            </ul>
          </div>
        )}
      </section>

      {/* Keywords */}
      <section>
        <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground mb-3">
          <Tag size={16} className="text-accent" />
          Keyword Analysis
        </h3>
        {data.keyword_analysis.present_keywords.length > 0 && (
          <div className="mb-3">
            <p className="text-xs font-medium text-muted-foreground mb-1.5">Present Keywords</p>
            <div className="flex flex-wrap gap-1.5">
              {data.keyword_analysis.present_keywords.map((kw) => (
                <span key={kw} className="rounded-full bg-success/10 px-2.5 py-0.5 text-xs font-medium text-success">
                  {kw}
                </span>
              ))}
            </div>
          </div>
        )}
        {data.keyword_analysis.missing_keywords.length > 0 && (
          <div className="mb-3">
            <p className="text-xs font-medium text-muted-foreground mb-1.5">Missing Keywords</p>
            <div className="flex flex-wrap gap-1.5">
              {data.keyword_analysis.missing_keywords.map((kw) => (
                <span key={kw} className="rounded-full bg-destructive/10 px-2.5 py-0.5 text-xs font-medium text-destructive">
                  {kw}
                </span>
              ))}
            </div>
          </div>
        )}
        {data.keyword_analysis.keyword_density_notes && (
          <p className="text-sm text-muted-foreground">{data.keyword_analysis.keyword_density_notes}</p>
        )}
      </section>

      {/* Section Feedback */}
      {Object.keys(data.section_feedback).length > 0 && (
        <section>
          <h3 className="text-sm font-semibold text-foreground mb-3">Section Feedback</h3>
          <div className="space-y-2">
            {Object.entries(data.section_feedback).map(([section, feedback]) => (
              <div key={section} className="rounded-lg border border-border bg-background p-3">
                <p className="text-xs font-semibold text-foreground capitalize mb-1">{section}</p>
                <p className="text-sm text-muted-foreground">{feedback}</p>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
