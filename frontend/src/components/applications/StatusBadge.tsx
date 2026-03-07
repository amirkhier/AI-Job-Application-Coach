import { cn } from '@/lib/cn';
import { STATUS_COLORS } from '@/lib/constants';
import type { ApplicationStatus } from '@/api/types';

interface StatusBadgeProps {
  status: ApplicationStatus;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium capitalize',
        STATUS_COLORS[status] ?? 'bg-muted text-muted-foreground',
        className,
      )}
    >
      {status}
    </span>
  );
}
