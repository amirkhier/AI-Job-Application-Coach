import { useState, type FormEvent } from 'react';
import { Plus } from 'lucide-react';
import { cn } from '@/lib/cn';

interface ApplicationFormProps {
  onSubmit: (data: {
    company_name: string;
    position_title: string;
    job_url?: string;
    notes?: string;
  }) => void;
  isLoading?: boolean;
  onCancel?: () => void;
}

export function ApplicationForm({ onSubmit, isLoading = false, onCancel }: ApplicationFormProps) {
  const [company, setCompany] = useState('');
  const [position, setPosition] = useState('');
  const [url, setUrl] = useState('');
  const [notes, setNotes] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!company.trim() || !position.trim()) return;
    onSubmit({
      company_name: company.trim(),
      position_title: position.trim(),
      job_url: url.trim() || undefined,
      notes: notes.trim() || undefined,
    });
    setCompany('');
    setPosition('');
    setUrl('');
    setNotes('');
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <label className="block text-sm font-medium text-foreground mb-1.5">Company *</label>
          <input
            type="text"
            value={company}
            onChange={(e) => setCompany(e.target.value)}
            placeholder="Company name"
            className="w-full rounded-xl border border-border bg-background px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-colors"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-foreground mb-1.5">Position *</label>
          <input
            type="text"
            value={position}
            onChange={(e) => setPosition(e.target.value)}
            placeholder="Position title"
            className="w-full rounded-xl border border-border bg-background px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-colors"
          />
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-foreground mb-1.5">
          Job URL <span className="text-muted-foreground font-normal">(optional)</span>
        </label>
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://..."
          className="w-full rounded-xl border border-border bg-background px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-colors"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-foreground mb-1.5">
          Notes <span className="text-muted-foreground font-normal">(optional)</span>
        </label>
        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Any notes about the application..."
          rows={2}
          className="w-full rounded-xl border border-border bg-background px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary resize-none transition-colors"
        />
      </div>

      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={isLoading || !company.trim() || !position.trim()}
          className={cn(
            'flex items-center gap-2 rounded-xl bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors',
            (isLoading || !company.trim() || !position.trim()) && 'opacity-50 cursor-not-allowed',
          )}
        >
          <Plus size={16} />
          {isLoading ? 'Adding...' : 'Add Application'}
        </button>
        {onCancel && (
          <button
            type="button"
            onClick={onCancel}
            className="rounded-xl border border-border px-6 py-2.5 text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
          >
            Cancel
          </button>
        )}
      </div>
    </form>
  );
}
