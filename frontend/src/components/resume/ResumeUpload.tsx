import { useState, type FormEvent } from 'react';
import { Upload, FileText, Loader2 } from 'lucide-react';
import { cn } from '@/lib/cn';
import { extractTextFromPdf, extractTextFromFile } from '@/lib/pdfExtract';

interface ResumeUploadProps {
  onSubmit: (resumeText: string, jobDescription?: string) => void;
  isLoading?: boolean;
  showJobDescription?: boolean;
}

export function ResumeUpload({ onSubmit, isLoading = false, showJobDescription = true }: ResumeUploadProps) {
  const [resumeText, setResumeText] = useState('');
  const [jobDescription, setJobDescription] = useState('');
  const [fileLoading, setFileLoading] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!resumeText.trim()) return;
    onSubmit(resumeText.trim(), jobDescription.trim() || undefined);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileError(null);
    setFileLoading(true);
    try {
      let text: string;
      if (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')) {
        text = await extractTextFromPdf(file);
      } else {
        text = await file.text();
      }
      if (!text.trim()) {
        setFileError('Could not extract text from this file. Try pasting your resume text directly.');
      } else {
        setResumeText(text);
      }
    } catch {
      setFileError('Failed to read file. Please try a different format or paste your text directly.');
    } finally {
      setFileLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-foreground mb-1.5">Resume Text</label>
        <div className="relative">
          <textarea
            value={resumeText}
            onChange={(e) => setResumeText(e.target.value)}
            placeholder="Paste your resume text here..."
            rows={8}
            className="w-full rounded-xl border border-border bg-background px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary resize-none transition-colors"
          />
          <label className={cn(
            'absolute bottom-3 right-3 cursor-pointer rounded-lg bg-muted p-2 text-muted-foreground hover:text-foreground transition-colors',
            fileLoading && 'pointer-events-none opacity-50',
          )}>
            {fileLoading ? <Loader2 size={16} className="animate-spin" /> : <Upload size={16} />}
            <input type="file" accept=".pdf,.txt,.md,.doc,.docx" onChange={handleFileUpload} className="sr-only" disabled={fileLoading} />
          </label>
        </div>
        {fileError && <p className="text-xs text-destructive mt-1">{fileError}</p>}
        <p className="text-xs text-muted-foreground mt-1">Minimum 50 characters. Upload a PDF or TXT file, or paste text directly.</p>
      </div>

      {showJobDescription && (
        <div>
          <label className="block text-sm font-medium text-foreground mb-1.5">
            Job Description <span className="text-muted-foreground font-normal">(optional)</span>
          </label>
          <textarea
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            placeholder="Paste the target job description for tailored analysis..."
            rows={4}
            className="w-full rounded-xl border border-border bg-background px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary resize-none transition-colors"
          />
        </div>
      )}

      <button
        type="submit"
        disabled={isLoading || resumeText.trim().length < 50}
        className={cn(
          'flex items-center gap-2 rounded-xl bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors',
          (isLoading || resumeText.trim().length < 50) && 'opacity-50 cursor-not-allowed',
        )}
      >
        <FileText size={16} />
        {isLoading ? 'Analyzing...' : 'Analyze Resume'}
      </button>
    </form>
  );
}
