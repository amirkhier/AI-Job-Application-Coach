import { Link } from 'react-router-dom';
import {
  MessageSquare,
  FileText,
  Mic,
  Search,
  ClipboardList,
  BookOpen,
  Activity,
  TrendingUp,
  ArrowRight,
} from 'lucide-react';
import { cn } from '@/lib/cn';
import { useHealth, useApplications, useUserProfile } from '@/hooks';
import { useAuthStore } from '@/stores';
import { ROUTES, STATUS_COLORS } from '@/lib/constants';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';

const quickActions = [
  { to: ROUTES.CHAT, label: 'Start Chat', icon: MessageSquare, color: 'text-primary', bg: 'bg-primary/10', desc: 'Ask anything about your career' },
  { to: ROUTES.RESUME, label: 'Analyze Resume', icon: FileText, color: 'text-success', bg: 'bg-success/10', desc: 'Get ATS score & feedback' },
  { to: ROUTES.INTERVIEW, label: 'Practice Interview', icon: Mic, color: 'text-warning', bg: 'bg-warning/10', desc: 'Mock interview with AI' },
  { to: ROUTES.JOBS, label: 'Search Jobs', icon: Search, color: 'text-accent', bg: 'bg-accent/10', desc: 'Find matching positions' },
  { to: ROUTES.APPLICATIONS, label: 'Track Applications', icon: ClipboardList, color: 'text-primary', bg: 'bg-primary/10', desc: 'Manage your pipeline' },
  { to: ROUTES.KNOWLEDGE, label: 'Career Tips', icon: BookOpen, color: 'text-success', bg: 'bg-success/10', desc: 'Expert career advice' },
];

function StatsCard({ label, value, icon: Icon, color }: { label: string; value: string | number; icon: React.ElementType; color: string }) {
  return (
    <div className="rounded-xl border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <p className="text-sm text-muted-foreground">{label}</p>
          <p className="text-2xl font-bold text-foreground">{value}</p>
        </div>
        <div className={cn('rounded-lg p-2.5', color)}>
          <Icon size={20} />
        </div>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const userId = useAuthStore((s) => s.userId);
  const { data: health } = useHealth();
  const { applications } = useApplications(userId);
  const { profile } = useUserProfile(userId);

  const appList = applications ?? [];
  const activeApps = appList.filter((a) => !['rejected', 'withdrawn'].includes(a.status));
  const interviewingCount = appList.filter((a) => a.status === 'interviewing').length;
  const offersCount = appList.filter((a) => a.status === 'offer').length;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Dashboard</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Welcome back! Here's an overview of your job search progress.
        </p>
      </div>

      {/* Stats */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatsCard
          label="Active Applications"
          value={activeApps.length}
          icon={ClipboardList}
          color="bg-primary/10 text-primary"
        />
        <StatsCard
          label="Interviewing"
          value={interviewingCount}
          icon={Mic}
          color="bg-warning/10 text-warning"
        />
        <StatsCard
          label="Offers"
          value={offersCount}
          icon={TrendingUp}
          color="bg-success/10 text-success"
        />
        <StatsCard
          label="System Status"
          value={health?.status ?? 'connecting'}
          icon={Activity}
          color={health?.status === 'healthy' ? 'bg-success/10 text-success' : 'bg-warning/10 text-warning'}
        />
      </div>

      {/* Quick Actions */}
      <div>
        <h2 className="text-lg font-semibold text-foreground mb-4">Quick Actions</h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {quickActions.map((action) => (
            <Link
              key={action.to}
              to={action.to}
              className="group flex items-center gap-4 rounded-xl border border-border bg-card p-4 hover:border-primary/30 hover:shadow-sm transition-all"
            >
              <div className={cn('rounded-lg p-2.5', action.bg, action.color)}>
                <action.icon size={20} />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-foreground">{action.label}</p>
                <p className="text-xs text-muted-foreground">{action.desc}</p>
              </div>
              <ArrowRight size={16} className="text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
            </Link>
          ))}
        </div>
      </div>

      {/* Recent Applications */}
      {appList.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-foreground">Recent Applications</h2>
            <Link to={ROUTES.APPLICATIONS} className="text-sm text-primary hover:underline">
              View all
            </Link>
          </div>
          <div className="rounded-xl border border-border bg-card divide-y divide-border overflow-hidden">
            {appList.slice(0, 5).map((app) => (
              <div key={app.id} className="flex items-center justify-between p-4">
                <div>
                  <p className="text-sm font-medium text-foreground">{app.position_title}</p>
                  <p className="text-xs text-muted-foreground">{app.company_name}</p>
                </div>
                <span className={cn('rounded-full px-2.5 py-0.5 text-xs font-medium', STATUS_COLORS[app.status])}>
                  {app.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Profile Summary */}
      {profile && profile.skills.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold text-foreground mb-4">Your Profile</h2>
          <div className="rounded-xl border border-border bg-card p-4 space-y-3">
            <div>
              <p className="text-xs text-muted-foreground">Experience Level</p>
              <p className="text-sm font-medium text-foreground capitalize">{profile.experience_level || 'Not set'}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Skills</p>
              <div className="flex flex-wrap gap-1.5 mt-1">
                {profile.skills.slice(0, 8).map((skill) => (
                  <span key={skill} className="rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary">
                    {skill}
                  </span>
                ))}
                {profile.skills.length > 8 && (
                  <span className="text-xs text-muted-foreground">+{profile.skills.length - 8} more</span>
                )}
              </div>
            </div>
            {profile.target_roles.length > 0 && (
              <div>
                <p className="text-xs text-muted-foreground">Target Roles</p>
                <p className="text-sm text-foreground">{profile.target_roles.join(', ')}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
