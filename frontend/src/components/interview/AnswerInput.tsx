import { useState, type FormEvent } from 'react';
import { Send } from 'lucide-react';
import { cn } from '@/lib/cn';

interface AnswerInputProps {
  onSubmit: (answer: string) => void;
  isLoading?: boolean;
}

export function AnswerInput({ onSubmit, isLoading = false }: AnswerInputProps) {
  const [answer, setAnswer] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!answer.trim() || isLoading) return;
    onSubmit(answer.trim());
    setAnswer('');
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <textarea
        value={answer}
        onChange={(e) => setAnswer(e.target.value)}
        placeholder="Type your answer here... (minimum 10 characters)"
        rows={5}
        className="w-full rounded-xl border border-border bg-background px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary resize-none transition-colors"
      />
      <button
        type="submit"
        disabled={isLoading || answer.trim().length < 10}
        className={cn(
          'flex items-center gap-2 rounded-xl bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors',
          (isLoading || answer.trim().length < 10) && 'opacity-50 cursor-not-allowed',
        )}
      >
        <Send size={16} />
        {isLoading ? 'Submitting...' : 'Submit Answer'}
      </button>
    </form>
  );
}
