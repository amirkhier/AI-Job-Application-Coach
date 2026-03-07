import { cn } from '@/lib/cn';

interface LoadingSkeletonProps {
  className?: string;
  lines?: number;
  variant?: 'text' | 'card' | 'circle';
}

function SkeletonLine({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        'h-4 animate-pulse rounded-md bg-muted',
        className,
      )}
    />
  );
}

export function LoadingSkeleton({ className, lines = 3, variant = 'text' }: LoadingSkeletonProps) {
  if (variant === 'circle') {
    return (
      <div className={cn('size-10 animate-pulse rounded-full bg-muted', className)} />
    );
  }

  if (variant === 'card') {
    return (
      <div className={cn('animate-pulse rounded-xl border border-border bg-card p-4 space-y-3', className)}>
        <SkeletonLine className="h-5 w-2/3" />
        <SkeletonLine className="h-4 w-full" />
        <SkeletonLine className="h-4 w-4/5" />
      </div>
    );
  }

  return (
    <div className={cn('space-y-2', className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <SkeletonLine
          key={i}
          className={i === lines - 1 ? 'w-3/5' : 'w-full'}
        />
      ))}
    </div>
  );
}

export function PageSkeleton() {
  return (
    <div className="space-y-6">
      <SkeletonLine className="h-8 w-48" />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {[1, 2, 3].map((i) => (
          <LoadingSkeleton key={i} variant="card" />
        ))}
      </div>
      <LoadingSkeleton lines={5} />
    </div>
  );
}
