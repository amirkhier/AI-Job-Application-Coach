import { useState } from 'react';
import { Search, MapPin, Building2 } from 'lucide-react';
import { useJobSearch } from '@/hooks';
import { useAuthStore } from '@/stores';
import { SearchForm, JobCard } from '@/components/jobs';
import { EmptyState } from '@/components/shared/EmptyState';
import { ErrorAlert } from '@/components/shared/ErrorAlert';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';
import { formatDuration } from '@/lib/formatters';
import type { JobSearchResponse } from '@/api/types';

export default function JobsPage() {
  const userId = useAuthStore((s) => s.userId);
  const { search, error, reset } = useJobSearch();
  const [results, setResults] = useState<JobSearchResponse | null>(null);

  const handleSearch = (query: string, location?: string, experienceLevel?: string, remoteOk?: boolean) => {
    search.mutate(
      { query, location, experience_level: experienceLevel, remote_ok: remoteOk, user_id: userId },
      { onSuccess: (data) => setResults(data) },
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Job Search</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Search for job listings matching your skills and preferences.
        </p>
      </div>

      {/* Search Form */}
      <div className="rounded-xl border border-border bg-card p-6">
        <SearchForm onSearch={handleSearch} isLoading={search.isPending} />
      </div>

      {/* Error */}
      {error && <ErrorAlert error={error} onRetry={reset} onDismiss={reset} />}

      {/* Loading */}
      {search.isPending && (
        <div className="grid gap-4 md:grid-cols-2">
          {[1, 2, 3, 4].map((i) => (
            <LoadingSkeleton key={i} variant="card" />
          ))}
        </div>
      )}

      {/* Results */}
      {!search.isPending && results && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              Found {results.total_found} jobs for "{results.search_query}"
              {results.location && ` in ${results.location}`}
              {results.processing_time > 0 && (
                <span className="ml-2">({formatDuration(results.processing_time)})</span>
              )}
            </p>
          </div>

          {/* Location info */}
          {results.location_info && (
            <div className="rounded-xl border border-border bg-card p-4">
              <div className="flex items-center gap-2 text-sm">
                <MapPin size={16} className="text-primary" />
                <span className="font-medium text-foreground">
                  {(results.location_info as { display_name?: string })?.display_name ?? results.location}
                </span>
              </div>
              {results.nearby_companies.length > 0 && (
                <div className="mt-3">
                  <p className="text-xs font-medium text-muted-foreground mb-2">Nearby Companies</p>
                  <div className="flex flex-wrap gap-2">
                    {results.nearby_companies.slice(0, 6).map((c, i) => (
                      <span key={i} className="flex items-center gap-1 rounded-full bg-muted px-2.5 py-0.5 text-xs text-foreground">
                        <Building2 size={10} />
                        {c.name}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Job listings */}
          {results.jobs.length > 0 ? (
            <div className="grid gap-4 md:grid-cols-2">
              {results.jobs.map((job, i) => (
                <JobCard key={i} job={job} />
              ))}
            </div>
          ) : (
            <EmptyState
              icon={<Search size={40} strokeWidth={1.5} />}
              title="No jobs found"
              description="Try adjusting your search criteria or location."
            />
          )}
        </div>
      )}

      {/* Initial state */}
      {!search.isPending && !results && !error && (
        <EmptyState
          icon={<Search size={40} strokeWidth={1.5} />}
          title="Search for Jobs"
          description="Enter a job title, skills, or keywords to find matching positions."
        />
      )}
    </div>
  );
}
