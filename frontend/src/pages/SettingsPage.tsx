import { useState } from 'react';
import { Settings, Moon, Sun, Monitor, Key, Trash2, Activity } from 'lucide-react';
import { cn } from '@/lib/cn';
import { useAuthStore, useUIStore, useChatStore } from '@/stores';
import { useHealth } from '@/hooks';

export default function SettingsPage() {
  const { apiKey, userId, setApiKey, setUserId, clearAuth } = useAuthStore();
  const { theme, setTheme, clearToasts } = useUIStore();
  const { data: health } = useHealth();
  const deleteAllSessions = useChatStore((s) => Object.keys(s.sessions).length);

  const [keyInput, setKeyInput] = useState(apiKey ?? '');
  const [userIdInput, setUserIdInput] = useState(String(userId));
  const [saved, setSaved] = useState(false);

  const handleSaveAuth = () => {
    setApiKey(keyInput.trim() || null);
    const uid = parseInt(userIdInput, 10);
    if (!isNaN(uid) && uid > 0) setUserId(uid);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const themes: { key: 'light' | 'dark' | 'system'; icon: React.ElementType; label: string }[] = [
    { key: 'light', icon: Sun, label: 'Light' },
    { key: 'dark', icon: Moon, label: 'Dark' },
    { key: 'system', icon: Monitor, label: 'System' },
  ];

  return (
    <div className="space-y-6 max-w-2xl">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Settings</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Configure your application preferences.
        </p>
      </div>

      {/* Theme */}
      <div className="rounded-xl border border-border bg-card p-6">
        <h2 className="text-lg font-semibold text-foreground mb-4">Appearance</h2>
        <div className="flex gap-3">
          {themes.map((t) => (
            <button
              key={t.key}
              onClick={() => setTheme(t.key)}
              className={cn(
                'flex flex-1 flex-col items-center gap-2 rounded-xl p-4 border transition-all',
                theme === t.key
                  ? 'border-primary bg-primary/5'
                  : 'border-border hover:border-primary/30',
              )}
            >
              <t.icon size={24} className={theme === t.key ? 'text-primary' : 'text-muted-foreground'} />
              <span className={cn('text-sm font-medium', theme === t.key ? 'text-primary' : 'text-muted-foreground')}>
                {t.label}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Authentication */}
      <div className="rounded-xl border border-border bg-card p-6">
        <h2 className="flex items-center gap-2 text-lg font-semibold text-foreground mb-4">
          <Key size={18} />
          Authentication
        </h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">API Key</label>
            <input
              type="password"
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              placeholder="Enter your API key..."
              className="w-full rounded-xl border border-border bg-background px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-colors"
            />
            <p className="text-xs text-muted-foreground mt-1">Used for authenticated API requests.</p>
          </div>
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">User ID</label>
            <input
              type="number"
              min={1}
              value={userIdInput}
              onChange={(e) => setUserIdInput(e.target.value)}
              className="w-full rounded-xl border border-border bg-background px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-colors"
            />
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={handleSaveAuth}
              className="flex items-center gap-2 rounded-xl bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              {saved ? 'Saved!' : 'Save'}
            </button>
            <button
              onClick={() => {
                clearAuth();
                setKeyInput('');
                setUserIdInput('1');
              }}
              className="rounded-xl border border-destructive/30 px-4 py-2.5 text-sm font-medium text-destructive hover:bg-destructive/5 transition-colors"
            >
              Clear Auth
            </button>
          </div>
        </div>
      </div>

      {/* System Health */}
      <div className="rounded-xl border border-border bg-card p-6">
        <h2 className="flex items-center gap-2 text-lg font-semibold text-foreground mb-4">
          <Activity size={18} />
          System Health
        </h2>
        {health ? (
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg bg-muted/50 p-3">
              <p className="text-xs text-muted-foreground">Status</p>
              <p className={cn('text-sm font-medium capitalize', health.status === 'healthy' ? 'text-success' : 'text-warning')}>
                {health.status}
              </p>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <p className="text-xs text-muted-foreground">Service</p>
              <p className="text-sm font-medium text-foreground">{health.service}</p>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <p className="text-xs text-muted-foreground">Database</p>
              <p className={cn('text-sm font-medium', health.checks.database ? 'text-success' : 'text-destructive')}>
                {health.checks.database ? 'Connected' : 'Disconnected'}
              </p>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <p className="text-xs text-muted-foreground">Redis</p>
              <p className={cn('text-sm font-medium', health.checks.redis ? 'text-success' : 'text-destructive')}>
                {health.checks.redis ? 'Connected' : 'Disconnected'}
              </p>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <p className="text-xs text-muted-foreground">ChromaDB</p>
              <p className={cn('text-sm font-medium', health.checks.chromadb ? 'text-success' : 'text-destructive')}>
                {health.checks.chromadb ? 'Connected' : 'Disconnected'}
              </p>
            </div>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">Connecting to backend...</p>
        )}
      </div>

      {/* Danger Zone */}
      <div className="rounded-xl border border-destructive/30 bg-card p-6">
        <h2 className="text-lg font-semibold text-destructive mb-4">Danger Zone</h2>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-foreground">Clear All Toasts</p>
              <p className="text-xs text-muted-foreground">Remove all notification toasts</p>
            </div>
            <button
              onClick={clearToasts}
              className="rounded-lg border border-destructive/30 px-3 py-1.5 text-sm text-destructive hover:bg-destructive/5 transition-colors"
            >
              Clear
            </button>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-foreground">Clear Chat Sessions</p>
              <p className="text-xs text-muted-foreground">{deleteAllSessions} session(s) stored</p>
            </div>
            <button
              onClick={() => {
                const store = useChatStore.getState();
                Object.keys(store.sessions).forEach((id) => store.deleteSession(id));
              }}
              className="flex items-center gap-1.5 rounded-lg border border-destructive/30 px-3 py-1.5 text-sm text-destructive hover:bg-destructive/5 transition-colors"
            >
              <Trash2 size={12} />
              Clear All
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
