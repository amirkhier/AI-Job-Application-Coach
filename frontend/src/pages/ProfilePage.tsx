import { useState, type FormEvent } from 'react';
import { User, Briefcase, Target, Star, Save, Plus, X, BarChart3 } from 'lucide-react';
import { cn } from '@/lib/cn';
import { useUserProfile } from '@/hooks';
import { useAuthStore } from '@/stores';
import { ErrorAlert } from '@/components/shared/ErrorAlert';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';
import { formatDate } from '@/lib/formatters';

export default function ProfilePage() {
  const userId = useAuthStore((s) => s.userId);
  const { profile, context, insights, updateProfile, isLoading, error } = useUserProfile(userId);

  const [isEditing, setIsEditing] = useState(false);
  const [skills, setSkills] = useState<string[]>([]);
  const [experienceLevel, setExperienceLevel] = useState('');
  const [experienceYears, setExperienceYears] = useState<number>(0);
  const [targetRoles, setTargetRoles] = useState<string[]>([]);
  const [careerGoals, setCareerGoals] = useState('');
  const [newSkill, setNewSkill] = useState('');
  const [newRole, setNewRole] = useState('');

  const startEditing = () => {
    if (profile) {
      setSkills([...profile.skills]);
      setExperienceLevel(profile.experience_level);
      setExperienceYears(profile.experience_years ?? 0);
      setTargetRoles([...profile.target_roles]);
      setCareerGoals(profile.career_goals);
    }
    setIsEditing(true);
  };

  const handleSave = (e: FormEvent) => {
    e.preventDefault();
    updateProfile.mutate(
      { skills, experience_level: experienceLevel, experience_years: experienceYears, target_roles: targetRoles, career_goals: careerGoals },
      { onSuccess: () => setIsEditing(false) },
    );
  };

  const addSkill = () => {
    if (newSkill.trim() && !skills.includes(newSkill.trim())) {
      setSkills([...skills, newSkill.trim()]);
      setNewSkill('');
    }
  };

  const addRole = () => {
    if (newRole.trim() && !targetRoles.includes(newRole.trim())) {
      setTargetRoles([...targetRoles, newRole.trim()]);
      setNewRole('');
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <LoadingSkeleton className="h-8 w-48" />
        <div className="grid gap-6 lg:grid-cols-2">
          <LoadingSkeleton variant="card" />
          <LoadingSkeleton variant="card" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Profile</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Manage your career profile and view personalized insights.
          </p>
        </div>
        {!isEditing && (
          <button
            onClick={startEditing}
            className="flex items-center gap-2 rounded-xl bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            Edit Profile
          </button>
        )}
      </div>

      {error && <ErrorAlert error={error} />}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Profile Card */}
        <div className="rounded-xl border border-border bg-card p-6">
          {isEditing ? (
            <form onSubmit={handleSave} className="space-y-4">
              <h2 className="text-lg font-semibold text-foreground">Edit Profile</h2>

              <div>
                <label className="block text-sm font-medium text-foreground mb-1.5">Experience Level</label>
                <select
                  value={experienceLevel}
                  onChange={(e) => setExperienceLevel(e.target.value)}
                  className="w-full rounded-xl border border-border bg-background px-4 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                >
                  <option value="">Select...</option>
                  <option value="junior">Junior</option>
                  <option value="mid">Mid</option>
                  <option value="senior">Senior</option>
                  <option value="lead">Lead</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-1.5">Years of Experience</label>
                <input
                  type="number"
                  min={0}
                  max={50}
                  value={experienceYears}
                  onChange={(e) => setExperienceYears(Number(e.target.value))}
                  className="w-full rounded-xl border border-border bg-background px-4 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-1.5">Skills</label>
                <div className="flex flex-wrap gap-1.5 mb-2">
                  {skills.map((skill) => (
                    <span key={skill} className="flex items-center gap-1 rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary">
                      {skill}
                      <button type="button" onClick={() => setSkills(skills.filter((s) => s !== skill))}>
                        <X size={10} />
                      </button>
                    </span>
                  ))}
                </div>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={newSkill}
                    onChange={(e) => setNewSkill(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addSkill())}
                    placeholder="Add skill..."
                    className="flex-1 rounded-lg border border-border bg-background px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                  />
                  <button type="button" onClick={addSkill} className="rounded-lg bg-muted p-1.5 text-muted-foreground hover:text-foreground">
                    <Plus size={14} />
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-1.5">Target Roles</label>
                <div className="flex flex-wrap gap-1.5 mb-2">
                  {targetRoles.map((role) => (
                    <span key={role} className="flex items-center gap-1 rounded-full bg-accent/10 px-2.5 py-0.5 text-xs font-medium text-accent">
                      {role}
                      <button type="button" onClick={() => setTargetRoles(targetRoles.filter((r) => r !== role))}>
                        <X size={10} />
                      </button>
                    </span>
                  ))}
                </div>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={newRole}
                    onChange={(e) => setNewRole(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addRole())}
                    placeholder="Add role..."
                    className="flex-1 rounded-lg border border-border bg-background px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                  />
                  <button type="button" onClick={addRole} className="rounded-lg bg-muted p-1.5 text-muted-foreground hover:text-foreground">
                    <Plus size={14} />
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-1.5">Career Goals</label>
                <textarea
                  value={careerGoals}
                  onChange={(e) => setCareerGoals(e.target.value)}
                  placeholder="Describe your career goals..."
                  rows={3}
                  className="w-full rounded-xl border border-border bg-background px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none transition-colors"
                />
              </div>

              <div className="flex items-center gap-3">
                <button
                  type="submit"
                  disabled={updateProfile.isPending}
                  className="flex items-center gap-2 rounded-xl bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
                >
                  <Save size={16} />
                  {updateProfile.isPending ? 'Saving...' : 'Save'}
                </button>
                <button
                  type="button"
                  onClick={() => setIsEditing(false)}
                  className="rounded-xl border border-border px-6 py-2.5 text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                >
                  Cancel
                </button>
              </div>
            </form>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="rounded-full bg-primary/10 p-3">
                  <User size={24} className="text-primary" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-foreground">User #{profile?.id ?? userId}</h2>
                  <p className="text-sm text-muted-foreground capitalize">
                    {profile?.experience_level || 'Not set'} Level
                    {profile?.experience_years ? ` · ${profile.experience_years} yrs` : ''}
                  </p>
                </div>
              </div>

              {profile?.skills && profile.skills.length > 0 && (
                <div>
                  <p className="flex items-center gap-1 text-xs font-medium text-muted-foreground mb-1.5">
                    <Star size={12} /> Skills
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {profile.skills.map((skill) => (
                      <span key={skill} className="rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary">
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {profile?.target_roles && profile.target_roles.length > 0 && (
                <div>
                  <p className="flex items-center gap-1 text-xs font-medium text-muted-foreground mb-1.5">
                    <Target size={12} /> Target Roles
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {profile.target_roles.map((role) => (
                      <span key={role} className="rounded-full bg-accent/10 px-2.5 py-0.5 text-xs font-medium text-accent">
                        {role}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {profile?.career_goals && (
                <div>
                  <p className="flex items-center gap-1 text-xs font-medium text-muted-foreground mb-1.5">
                    <Briefcase size={12} /> Career Goals
                  </p>
                  <p className="text-sm text-foreground">{profile.career_goals}</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Insights */}
        <div className="space-y-4">
          {/* Context summary */}
          {context && (
            <div className="rounded-xl border border-border bg-card p-5">
              <h3 className="text-sm font-semibold text-foreground mb-3">Activity Summary</h3>
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-lg bg-muted/50 p-3 text-center">
                  <p className="text-2xl font-bold text-foreground">{context.context_summary?.total_conversations ?? 0}</p>
                  <p className="text-xs text-muted-foreground">Conversations</p>
                </div>
                <div className="rounded-lg bg-muted/50 p-3 text-center">
                  <p className="text-2xl font-bold text-foreground">{context.history_count}</p>
                  <p className="text-xs text-muted-foreground">History Items</p>
                </div>
              </div>
              {'text' in context.context_summary && context.context_summary.text && (
                <div className="mt-3">
                  <p className="text-xs text-muted-foreground mb-1.5">Summary</p>
                  <p className="text-sm text-foreground">{context.context_summary.text}</p>
                </div>
              )}
              {(context.context_summary?.top_intents?.length ?? 0) > 0 && (
                <div className="mt-3">
                  <p className="text-xs text-muted-foreground mb-1.5">Top Intents</p>
                  <div className="flex flex-wrap gap-1.5">
                    {context.context_summary.top_intents.map((intent: string) => (
                      <span key={intent} className="rounded-full bg-primary/10 px-2.5 py-0.5 text-xs text-primary">
                        {intent}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Insights */}
          {insights && (
            <div className="rounded-xl border border-border bg-card p-5">
              <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground mb-3">
                <BarChart3 size={14} /> Conversation Insights
              </h3>
              {(insights.insights?.length ?? 0) > 0 && (
                <div className="mb-3">
                  <p className="text-xs text-muted-foreground mb-1.5">Key Insights</p>
                  <ul className="space-y-1">
                    {insights.insights.map((insight: string, i: number) => (
                      <li key={i} className="text-sm text-foreground">{insight}</li>
                    ))}
                  </ul>
                </div>
              )}
              {(insights.recommendations?.length ?? 0) > 0 && (
                <div>
                  <p className="text-xs text-muted-foreground mb-1.5">Recommendations</p>
                  <ul className="space-y-1">
                    {insights.recommendations.map((rec: string, i: number) => (
                      <li key={i} className="text-sm text-foreground">{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
              {Object.keys(insights.agent_usage ?? {}).length > 0 && (
                <div className="mt-3">
                  <p className="text-xs text-muted-foreground mb-1.5">Agent Usage</p>
                  <div className="space-y-1.5">
                    {Object.entries(insights.agent_usage)
                      .sort(([, a], [, b]) => b - a)
                      .map(([agent, count]) => (
                        <div key={agent} className="flex items-center justify-between">
                          <span className="text-xs text-foreground capitalize">{agent}</span>
                          <div className="flex items-center gap-2">
                            <div className="h-1.5 w-24 rounded-full bg-muted overflow-hidden">
                              <div
                                className="h-full rounded-full bg-primary"
                                style={{ width: `${(count / Math.max(...Object.values(insights.agent_usage))) * 100}%` }}
                              />
                            </div>
                            <span className="text-xs text-muted-foreground w-6 text-right">{count}</span>
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Recent conversations */}
          {context?.recent_conversations && context.recent_conversations.length > 0 && (
            <div className="rounded-xl border border-border bg-card p-5">
              <h3 className="text-sm font-semibold text-foreground mb-3">Recent Conversations</h3>
              <div className="space-y-2">
                {context.recent_conversations.slice(0, 5).map((conv: any, i: number) => (
                  <div key={i} className="rounded-lg bg-muted/50 p-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-primary capitalize">{conv.intent ?? conv.agent_used ?? ''}</span>
                      <span className="text-[10px] text-muted-foreground">{formatDate(conv.created_at ?? conv.timestamp)}</span>
                    </div>
                    <p className="text-sm text-foreground">{conv.metadata?.summary ?? conv.summary ?? conv.message ?? ''}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
