import { useState } from 'react';
import { Plus, ClipboardList, Bell, Filter } from 'lucide-react';
import { cn } from '@/lib/cn';
import { useApplications } from '@/hooks';
import { useAuthStore } from '@/stores';
import { ApplicationForm, ApplicationTable, StatusBadge } from '@/components/applications';
import { EmptyState } from '@/components/shared/EmptyState';
import { ErrorAlert } from '@/components/shared/ErrorAlert';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';
import type { ApplicationStatus } from '@/api/types';

export default function ApplicationsPage() {
  const userId = useAuthStore((s) => s.userId);
  const [showForm, setShowForm] = useState(false);
  const [statusFilter, setStatusFilter] = useState<string>('');

  const {
    applications,
    followUps,
    isLoading,
    error,
    create,
    update,
    remove,
  } = useApplications(userId);

  const filteredApps = statusFilter
    ? (applications ?? []).filter((a) => a.status === statusFilter)
    : (applications ?? []);

  const statuses: ApplicationStatus[] = ['applied', 'interviewing', 'offer', 'rejected', 'withdrawn'];
  const statusCounts = statuses.reduce(
    (acc, s) => ({ ...acc, [s]: (applications ?? []).filter((a) => a.status === s).length }),
    {} as Record<string, number>,
  );

  const handleCreate = (data: { company_name: string; position_title: string; job_url?: string; notes?: string }) => {
    create.mutate(
      { ...data, user_id: userId },
      { onSuccess: () => setShowForm(false) },
    );
  };

  const handleUpdateStatus = (id: number, status: ApplicationStatus) => {
    update.mutate({ id, data: { status } });
  };

  const handleDelete = (id: number) => {
    remove.mutate(id);
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <LoadingSkeleton className="h-8 w-48" />
        <div className="grid gap-3 sm:grid-cols-5">
          {[1, 2, 3, 4, 5].map((i) => <LoadingSkeleton key={i} variant="card" />)}
        </div>
        <LoadingSkeleton lines={8} />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Application Tracker</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Track and manage your job applications across the pipeline.
          </p>
        </div>
        <button
          onClick={() => setShowForm(!showForm)}
          className={cn(
            'flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium transition-colors',
            showForm
              ? 'border border-border text-muted-foreground hover:text-foreground'
              : 'bg-primary text-primary-foreground hover:bg-primary/90',
          )}
        >
          <Plus size={16} />
          {showForm ? 'Cancel' : 'Add Application'}
        </button>
      </div>

      {error && <ErrorAlert error={error} />}

      {/* Add form */}
      {showForm && (
        <div className="rounded-xl border border-border bg-card p-6">
          <h2 className="text-lg font-semibold text-foreground mb-4">New Application</h2>
          <ApplicationForm
            onSubmit={handleCreate}
            isLoading={create.isPending}
            onCancel={() => setShowForm(false)}
          />
        </div>
      )}

      {/* Pipeline stats */}
      <div className="grid gap-3 sm:grid-cols-5">
        {statuses.map((status) => (
          <button
            key={status}
            onClick={() => setStatusFilter(statusFilter === status ? '' : status)}
            className={cn(
              'rounded-xl border p-3 text-center transition-all',
              statusFilter === status
                ? 'border-primary bg-primary/5'
                : 'border-border bg-card hover:border-primary/30',
            )}
          >
            <StatusBadge status={status} />
            <p className="mt-2 text-2xl font-bold text-foreground">{statusCounts[status]}</p>
          </button>
        ))}
      </div>

      {/* Follow-ups */}
      {followUps && followUps.length > 0 && (
        <div className="rounded-xl border border-warning/30 bg-warning/5 p-4">
          <div className="flex items-center gap-2 mb-2">
            <Bell size={16} className="text-warning" />
            <span className="text-sm font-semibold text-foreground">
              {followUps.length} Follow-up{followUps.length > 1 ? 's' : ''} Due
            </span>
          </div>
          <div className="space-y-1">
            {followUps.slice(0, 3).map((app) => (
              <p key={app.id} className="text-sm text-muted-foreground">
                {app.position_title} at {app.company_name}
              </p>
            ))}
          </div>
        </div>
      )}

      {/* Table */}
      {filteredApps.length > 0 ? (
        <div>
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm text-muted-foreground">
              {statusFilter ? (
                <>Showing {filteredApps.length} {statusFilter} application{filteredApps.length !== 1 ? 's' : ''}</>
              ) : (
                <>{filteredApps.length} application{filteredApps.length !== 1 ? 's' : ''}</>
              )}
            </p>
            {statusFilter && (
              <button
                onClick={() => setStatusFilter('')}
                className="flex items-center gap-1 text-xs text-primary hover:underline"
              >
                <Filter size={12} /> Clear filter
              </button>
            )}
          </div>
          <ApplicationTable
            applications={filteredApps}
            onUpdateStatus={handleUpdateStatus}
            onDelete={handleDelete}
            isUpdating={update.isPending}
          />
        </div>
      ) : (
        <EmptyState
          icon={<ClipboardList size={40} strokeWidth={1.5} />}
          title={statusFilter ? `No ${statusFilter} applications` : 'No Applications Yet'}
          description={statusFilter ? 'Try a different filter.' : 'Start tracking your job applications by adding your first one.'}
          action={
            !statusFilter && !showForm ? (
              <button
                onClick={() => setShowForm(true)}
                className="flex items-center gap-2 rounded-xl bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
              >
                <Plus size={16} />
                Add Application
              </button>
            ) : undefined
          }
        />
      )}
    </div>
  );
}
