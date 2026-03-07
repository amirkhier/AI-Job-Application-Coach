import { NavLink, useLocation } from 'react-router-dom';
import {
  MessageSquare,
  FileText,
  Mic,
  Search,
  BookOpen,
  ClipboardList,
  User,
  Settings,
  LayoutDashboard,
  ChevronLeft,
} from 'lucide-react';
import { cn } from '@/lib/cn';
import { ROUTES } from '@/lib/constants';
import { useUIStore } from '@/stores';

const navItems = [
  { to: ROUTES.DASHBOARD, label: 'Dashboard', icon: LayoutDashboard },
  { to: ROUTES.CHAT, label: 'Chat', icon: MessageSquare },
  { to: ROUTES.RESUME, label: 'Resume', icon: FileText },
  { to: ROUTES.INTERVIEW, label: 'Interview', icon: Mic },
  { to: ROUTES.JOBS, label: 'Jobs', icon: Search },
  { to: ROUTES.KNOWLEDGE, label: 'Knowledge', icon: BookOpen },
  { to: ROUTES.APPLICATIONS, label: 'Applications', icon: ClipboardList },
];

const bottomItems = [
  { to: ROUTES.PROFILE, label: 'Profile', icon: User },
  { to: ROUTES.SETTINGS, label: 'Settings', icon: Settings },
];

export function Sidebar() {
  const sidebarOpen = useUIStore((s) => s.sidebarOpen);
  const toggleSidebar = useUIStore((s) => s.toggleSidebar);
  const location = useLocation();

  return (
    <aside
      className={cn(
        'fixed inset-y-0 left-0 z-30 flex flex-col border-r border-border bg-card transition-all duration-300',
        sidebarOpen ? 'w-60' : 'w-16',
      )}
    >
      {/* Header */}
      <div className="flex h-14 items-center justify-between border-b border-border px-3">
        {sidebarOpen && (
          <span className="text-sm font-semibold text-foreground truncate">
            AI Job Coach
          </span>
        )}
        <button
          onClick={toggleSidebar}
          className="rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
          aria-label="Toggle sidebar"
        >
          <ChevronLeft
            size={18}
            className={cn('transition-transform', !sidebarOpen && 'rotate-180')}
          />
        </button>
      </div>

      {/* Main nav */}
      <nav className="flex-1 overflow-y-auto py-2 px-2 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary/10 text-primary'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground',
                !sidebarOpen && 'justify-center px-2',
              )
            }
          >
            <item.icon size={18} />
            {sidebarOpen && <span>{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Bottom nav */}
      <div className="border-t border-border py-2 px-2 space-y-1">
        {bottomItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary/10 text-primary'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground',
                !sidebarOpen && 'justify-center px-2',
              )
            }
          >
            <item.icon size={18} />
            {sidebarOpen && <span>{item.label}</span>}
          </NavLink>
        ))}
      </div>
    </aside>
  );
}
