import { useState } from 'react';
import { FileText, Wand2, ShieldCheck } from 'lucide-react';
import { cn } from '@/lib/cn';
import { useResumeAnalysis } from '@/hooks';
import { useAuthStore } from '@/stores';
import { ResumeUpload, ScoreCard, ResumeResults, ImprovementResults } from '@/components/resume';
import { ErrorAlert } from '@/components/shared/ErrorAlert';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';
import { formatDuration } from '@/lib/formatters';
import type { ResumeResponse, ResumeImprovementResponse } from '@/api/types';

type Tab = 'analyze' | 'improve' | 'audit';

export default function ResumePage() {
  const userId = useAuthStore((s) => s.userId);
  const [activeTab, setActiveTab] = useState<Tab>('analyze');
  const [analyzeResult, setAnalyzeResult] = useState<ResumeResponse | null>(null);
  const [improveResult, setImproveResult] = useState<ResumeImprovementResponse | null>(null);
  const [auditResult, setAuditResult] = useState<ResumeResponse | null>(null);

  const { analyze, improve, audit } = useResumeAnalysis();

  const isLoading = analyze.isPending || improve.isPending || audit.isPending;
  const error = analyze.error || improve.error || audit.error;

  const handleAnalyze = (resumeText: string, jobDescription?: string) => {
    analyze.mutate(
      { resume_text: resumeText, job_description: jobDescription, user_id: userId },
      { onSuccess: (data) => setAnalyzeResult(data) },
    );
  };

  const handleImprove = (resumeText: string, jobDescription?: string) => {
    improve.mutate(
      { resume_text: resumeText, job_description: jobDescription, user_id: userId },
      { onSuccess: (data) => setImproveResult(data) },
    );
  };

  const handleAudit = (resumeText: string, jobDescription?: string) => {
    audit.mutate(
      { resume_text: resumeText, job_description: jobDescription, user_id: userId },
      { onSuccess: (data) => setAuditResult(data) },
    );
  };

  const tabs = [
    { key: 'analyze' as Tab, label: 'Analyze', icon: FileText },
    { key: 'improve' as Tab, label: 'Improve', icon: Wand2 },
    { key: 'audit' as Tab, label: 'ATS Audit', icon: ShieldCheck },
  ];

  const handlers = { analyze: handleAnalyze, improve: handleImprove, audit: handleAudit };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Resume Analysis</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Get AI-powered feedback on your resume with ATS scoring, improvements, and keyword analysis.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 rounded-xl border border-border bg-card p-1">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={cn(
              'flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors flex-1',
              activeTab === tab.key
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted',
            )}
          >
            <tab.icon size={16} />
            {tab.label}
          </button>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Input */}
        <div className="rounded-xl border border-border bg-card p-6">
          <h2 className="text-lg font-semibold text-foreground mb-4">
            {activeTab === 'analyze' && 'Full Resume Analysis'}
            {activeTab === 'improve' && 'Resume Improvement'}
            {activeTab === 'audit' && 'ATS Compliance Audit'}
          </h2>
          <ResumeUpload
            onSubmit={handlers[activeTab]}
            isLoading={isLoading}
            showJobDescription={activeTab !== 'audit'}
          />
        </div>

        {/* Results */}
        <div className="rounded-xl border border-border bg-card p-6">
          <h2 className="text-lg font-semibold text-foreground mb-4">Results</h2>

          {error && <ErrorAlert error={error} onRetry={() => {}} />}

          {isLoading && <LoadingSkeleton lines={8} />}

          {!isLoading && activeTab === 'analyze' && analyzeResult && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <ScoreCard label="Overall Score" score={analyzeResult.overall_score} />
                <ScoreCard label="ATS Score" score={analyzeResult.ats_compatibility.score} />
              </div>
              <p className="text-xs text-muted-foreground">
                Processed in {formatDuration(analyzeResult.processing_time)}
              </p>
              <ResumeResults data={analyzeResult} />
            </div>
          )}

          {!isLoading && activeTab === 'improve' && improveResult && (
            <div className="space-y-4">
              <p className="text-xs text-muted-foreground">
                Processed in {formatDuration(improveResult.processing_time)}
              </p>
              <ImprovementResults data={improveResult} />
            </div>
          )}

          {!isLoading && activeTab === 'audit' && auditResult && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <ScoreCard label="Overall Score" score={auditResult.overall_score} />
                <ScoreCard label="ATS Score" score={auditResult.ats_compatibility.score} />
              </div>
              <p className="text-xs text-muted-foreground">
                Processed in {formatDuration(auditResult.processing_time)}
              </p>
              <ResumeResults data={auditResult} />
            </div>
          )}

          {!isLoading && !error && activeTab === 'analyze' && !analyzeResult && (
            <p className="text-sm text-muted-foreground text-center py-8">Submit your resume to see analysis results.</p>
          )}
          {!isLoading && !error && activeTab === 'improve' && !improveResult && (
            <p className="text-sm text-muted-foreground text-center py-8">Submit your resume to see improvement suggestions.</p>
          )}
          {!isLoading && !error && activeTab === 'audit' && !auditResult && (
            <p className="text-sm text-muted-foreground text-center py-8">Submit your resume to see the ATS audit.</p>
          )}
        </div>
      </div>
    </div>
  );
}
