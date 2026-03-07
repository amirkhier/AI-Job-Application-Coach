import { useState, type FormEvent } from 'react';
import { Play } from 'lucide-react';
import { cn } from '@/lib/cn';
import { INTERVIEW_LEVELS } from '@/lib/constants';

interface SessionSetupProps {
  onStart: (role: string, level: string, questionCount: number) => void;
  isLoading?: boolean;
}

export function SessionSetup({ onStart, isLoading = false }: SessionSetupProps) {
  const [role, setRole] = useState('');
  const [level, setLevel] = useState<string>('mid');
  const [questionCount, setQuestionCount] = useState(5);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!role.trim()) return;
    onStart(role.trim(), level, questionCount);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <div>
        <label className="block text-sm font-medium text-foreground mb-1.5">Target Role</label>
        <input
          type="text"
          value={role}
          onChange={(e) => setRole(e.target.value)}
          placeholder="e.g., Frontend Developer, Product Manager..."
          className="w-full rounded-xl border border-border bg-background px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-colors"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-foreground mb-1.5">Experience Level</label>
        <div className="flex gap-2">
          {INTERVIEW_LEVELS.map((l) => (
            <button
              key={l}
              type="button"
              onClick={() => setLevel(l)}
              className={cn(
                'flex-1 rounded-lg px-3 py-2 text-sm font-medium transition-colors capitalize',
                level === l
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-muted-foreground hover:text-foreground',
              )}
            >
              {l}
            </button>
          ))}
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-foreground mb-1.5">
          Number of Questions: {questionCount}
        </label>
        <input
          type="range"
          min={1}
          max={10}
          value={questionCount}
          onChange={(e) => setQuestionCount(Number(e.target.value))}
          className="w-full accent-primary"
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>1</span>
          <span>10</span>
        </div>
      </div>

      <button
        type="submit"
        disabled={isLoading || !role.trim()}
        className={cn(
          'flex items-center gap-2 rounded-xl bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors w-full justify-center',
          (isLoading || !role.trim()) && 'opacity-50 cursor-not-allowed',
        )}
      >
        <Play size={16} />
        {isLoading ? 'Starting...' : 'Start Interview'}
      </button>
    </form>
  );
}
