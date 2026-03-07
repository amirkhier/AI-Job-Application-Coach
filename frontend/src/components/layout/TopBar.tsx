import { Moon, Sun, Monitor, Activity } from 'lucide-react';
import { cn } from '@/lib/cn';
import { useUIStore } from '@/stores';
import { useHealth } from '@/hooks';

export function TopBar() {
  const theme = useUIStore((s) => s.theme);
  const setTheme = useUIStore((s) => s.setTheme);
  const sidebarOpen = useUIStore((s) => s.sidebarOpen);
  const { data: health } = useHealth();

  const cycleTheme = () => {
    const order: Array<'light' | 'dark' | 'system'> = ['light', 'dark', 'system'];
    const next = order[(order.indexOf(theme) + 1) % order.length];
    setTheme(next);
  };

  const ThemeIcon = theme === 'dark' ? Moon : theme === 'light' ? Sun : Monitor;

  return (
    <header
      className={cn(
        'fixed top-0 right-0 z-20 flex h-14 items-center justify-between border-b border-border bg-card/80 backdrop-blur-sm px-4 transition-all duration-300',
        sidebarOpen ? 'left-60' : 'left-16',
      )}
    >
      <div className="text-sm font-medium text-foreground">
        AI Job Application Coach
      </div>

      <div className="flex items-center gap-3">
        {/* Health indicator */}
        <div
          className={cn(
            'flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium',
            health?.status === 'healthy'
              ? 'bg-success/10 text-success'
              : health
                ? 'bg-warning/10 text-warning'
                : 'bg-muted text-muted-foreground',
          )}
        >
          <Activity size={12} />
          <span>{health?.status ?? 'connecting'}</span>
        </div>

        {/* Theme toggle */}
        <button
          onClick={cycleTheme}
          className="rounded-md p-2 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
          aria-label={`Switch theme (current: ${theme})`}
        >
          <ThemeIcon size={16} />
        </button>
      </div>
    </header>
  );
}
