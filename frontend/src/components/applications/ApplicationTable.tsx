import { ExternalLink, Trash2, Calendar, ChevronDown } from 'lucide-react';
import { useState } from 'react';
import { cn } from '@/lib/cn';
import { StatusBadge } from './StatusBadge';
import { STATUS_TRANSITIONS } from '@/lib/constants';
import { formatDate } from '@/lib/formatters';
import type { ApplicationResponse, ApplicationStatus } from '@/api/types';

interface ApplicationTableProps {
  applications: ApplicationResponse[];
  onUpdateStatus: (id: number, status: ApplicationStatus) => void;
  onDelete: (id: number) => void;
  isUpdating?: boolean;
}

export function ApplicationTable({ applications, onUpdateStatus, onDelete, isUpdating }: ApplicationTableProps) {
  const [expandedId, setExpandedId] = useState<number | null>(null);

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      <table className="w-full">
        <thead>
          <tr className="border-b border-border bg-muted/50">
            <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground">Position</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground">Company</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground">Status</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground">Applied</th>
            <th className="px-4 py-3 text-right text-xs font-medium text-muted-foreground">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {applications.map((app) => {
            const validTransitions = STATUS_TRANSITIONS[app.status] ?? [];
            const isExpanded = expandedId === app.id;

            return (
              <tr
                key={app.id}
                className="group hover:bg-muted/30 transition-colors"
              >
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setExpandedId(isExpanded ? null : app.id)}
                      className="text-muted-foreground hover:text-foreground transition-colors"
                    >
                      <ChevronDown size={14} className={cn('transition-transform', isExpanded && 'rotate-180')} />
                    </button>
                    <div>
                      <p className="text-sm font-medium text-foreground">{app.position_title}</p>
                      {isExpanded && app.notes && (
                        <p className="text-xs text-muted-foreground mt-1">{app.notes}</p>
                      )}
                    </div>
                  </div>
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-foreground">{app.company_name}</span>
                    {app.job_url && (
                      <a
                        href={app.job_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-muted-foreground hover:text-primary transition-colors"
                      >
                        <ExternalLink size={12} />
                      </a>
                    )}
                  </div>
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <StatusBadge status={app.status} />
                    {validTransitions.length > 0 && (
                      <select
                        onChange={(e) => {
                          if (e.target.value) {
                            onUpdateStatus(app.id, e.target.value as ApplicationStatus);
                            e.target.value = '';
                          }
                        }}
                        disabled={isUpdating}
                        className="rounded border border-border bg-background px-1.5 py-0.5 text-xs text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                        defaultValue=""
                      >
                        <option value="" disabled>Move to...</option>
                        {validTransitions.map((s) => (
                          <option key={s} value={s} className="capitalize">{s}</option>
                        ))}
                      </select>
                    )}
                  </div>
                </td>
                <td className="px-4 py-3">
                  <div className="space-y-0.5">
                    <p className="text-sm text-foreground">{formatDate(app.application_date)}</p>
                    {app.follow_up_date && (
                      <p className="flex items-center gap-1 text-xs text-warning">
                        <Calendar size={10} />
                        Follow up: {formatDate(app.follow_up_date)}
                      </p>
                    )}
                  </div>
                </td>
                <td className="px-4 py-3 text-right">
                  <button
                    onClick={() => onDelete(app.id)}
                    className="rounded-md p-1.5 text-muted-foreground opacity-0 group-hover:opacity-100 hover:bg-destructive/10 hover:text-destructive transition-all"
                    aria-label="Delete"
                  >
                    <Trash2 size={14} />
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
