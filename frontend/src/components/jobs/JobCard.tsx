import { MapPin, Building2, ExternalLink, Star, Wifi } from 'lucide-react';
import { cn } from '@/lib/cn';
import type { JobListing } from '@/api/types';
import { formatScore } from '@/lib/formatters';

interface JobCardProps {
  job: JobListing;
}

export function JobCard({ job }: JobCardProps) {
  return (
    <div className="rounded-xl border border-border bg-card p-5 hover:border-primary/30 hover:shadow-sm transition-all">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0 space-y-1">
          <h3 className="text-base font-semibold text-foreground truncate">{job.title}</h3>
          <div className="flex items-center gap-3 text-sm text-muted-foreground">
            <span className="flex items-center gap-1">
              <Building2 size={14} />
              {job.company}
            </span>
            <span className="flex items-center gap-1">
              <MapPin size={14} />
              {job.location}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {job.remote_friendly && (
            <span className="flex items-center gap-1 rounded-full bg-success/10 px-2.5 py-0.5 text-xs font-medium text-success">
              <Wifi size={10} />
              Remote
            </span>
          )}
          {job.match_score != null && (
            <span
              className={cn(
                'flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium',
                job.match_score >= 8
                  ? 'bg-success/10 text-success'
                  : job.match_score >= 6
                    ? 'bg-warning/10 text-warning'
                    : 'bg-muted text-muted-foreground',
              )}
            >
              <Star size={10} />
              {formatScore(job.match_score)}
            </span>
          )}
        </div>
      </div>

      <p className="mt-3 text-sm text-muted-foreground line-clamp-3">{job.description}</p>

      <div className="mt-3 flex flex-wrap gap-1.5">
        {job.key_skills.slice(0, 5).map((skill) => (
          <span key={skill} className="rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary">
            {skill}
          </span>
        ))}
        {job.key_skills.length > 5 && (
          <span className="text-xs text-muted-foreground">+{job.key_skills.length - 5}</span>
        )}
      </div>

      <div className="mt-3 flex items-center justify-between">
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          {job.salary_range && <span>{job.salary_range}</span>}
          <span className="capitalize">{job.experience_level}</span>
        </div>
        {job.url && (
          <a
            href={job.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-xs font-medium text-primary hover:underline"
          >
            View <ExternalLink size={12} />
          </a>
        )}
      </div>
    </div>
  );
}
