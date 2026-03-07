import { useState, type FormEvent } from 'react';
import { BookOpen, Search, Link2, Tag, Star } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { cn } from '@/lib/cn';
import { useKnowledge } from '@/hooks';
import { useAuthStore } from '@/stores';
import { EmptyState } from '@/components/shared/EmptyState';
import { ErrorAlert } from '@/components/shared/ErrorAlert';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';
import { formatScore } from '@/lib/formatters';
import type { KnowledgeQueryResponse } from '@/api/types';

export default function KnowledgePage() {
  const userId = useAuthStore((s) => s.userId);
  const [query, setQuery] = useState('');
  const [result, setResult] = useState<KnowledgeQueryResponse | null>(null);

  const { ask, error, reset } = useKnowledge();

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim() || ask.isPending) return;
    ask.mutate(
      { query: query.trim(), user_id: userId },
      { onSuccess: (data) => setResult(data) },
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Knowledge Base</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Get expert career advice powered by our curated knowledge base.
        </p>
      </div>

      {/* Query form */}
      <form onSubmit={handleSubmit} className="flex gap-3">
        <div className="relative flex-1">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about resume tips, interview prep, salary negotiation..."
            className="w-full rounded-xl border border-border bg-card pl-10 pr-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-colors"
          />
        </div>
        <button
          type="submit"
          disabled={ask.isPending || query.trim().length < 5}
          className={cn(
            'flex items-center gap-2 rounded-xl bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors',
            (ask.isPending || query.trim().length < 5) && 'opacity-50 cursor-not-allowed',
          )}
        >
          <BookOpen size={16} />
          {ask.isPending ? 'Searching...' : 'Ask'}
        </button>
      </form>

      {error && <ErrorAlert error={error} onRetry={reset} onDismiss={reset} />}

      {ask.isPending && (
        <div className="rounded-xl border border-border bg-card p-6">
          <LoadingSkeleton lines={6} />
        </div>
      )}

      {/* Results */}
      {!ask.isPending && result && (
        <div className="space-y-4">
          {/* Answer */}
          <div className="rounded-xl border border-border bg-card p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-foreground">Answer</h2>
              <span
                className={cn(
                  'flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium',
                  result.relevance_score >= 0.8
                    ? 'bg-success/10 text-success'
                    : result.relevance_score >= 0.5
                      ? 'bg-warning/10 text-warning'
                      : 'bg-muted text-muted-foreground',
                )}
              >
                <Star size={10} />
                Relevance: {formatScore(result.relevance_score * 10)}/10
              </span>
            </div>
            <div className="prose prose-sm max-w-none dark:prose-invert">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.answer}</ReactMarkdown>
            </div>
          </div>

          {/* Sources */}
          {result.sources.length > 0 && (
            <div className="rounded-xl border border-border bg-card p-4">
              <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground mb-2">
                <Link2 size={14} />
                Sources
              </h3>
              <div className="flex flex-wrap gap-2">
                {result.sources.map((source, i) => (
                  <span key={i} className="rounded-full bg-muted px-3 py-1 text-xs text-muted-foreground">
                    {source}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Related Topics */}
          {result.related_topics.length > 0 && (
            <div className="rounded-xl border border-border bg-card p-4">
              <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground mb-2">
                <Tag size={14} />
                Related Topics
              </h3>
              <div className="flex flex-wrap gap-2">
                {result.related_topics.map((topic) => (
                  <button
                    key={topic}
                    onClick={() => {
                      setQuery(topic);
                    }}
                    className="rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary hover:bg-primary/20 transition-colors cursor-pointer"
                  >
                    {topic}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Initial state */}
      {!ask.isPending && !result && !error && (
        <EmptyState
          icon={<BookOpen size={40} strokeWidth={1.5} />}
          title="Ask a Career Question"
          description="Our knowledge base covers resume writing, interview preparation, salary negotiation, and industry insights."
        />
      )}
    </div>
  );
}
