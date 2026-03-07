import { Mic } from 'lucide-react';
import { useInterviewStore, useAuthStore } from '@/stores';
import { useInterview } from '@/hooks';
import { SessionSetup, QuestionCard, AnswerInput, FeedbackPanel, SessionSummaryCard } from '@/components/interview';
import { ErrorAlert } from '@/components/shared/ErrorAlert';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';

export default function InterviewPage() {
  const userId = useAuthStore((s) => s.userId);
  const {
    sessionId,
    role,
    level,
    currentQuestion,
    answers,
    totalQuestions,
    isComplete,
    summary,
    isSubmitting,
    resetSession,
  } = useInterviewStore();

  const { startInterview, submitAnswer, error, reset } = useInterview();

  const handleStart = (role: string, level: string, questionCount: number) => {
    startInterview({ role, level, question_count: questionCount, user_id: userId });
  };

  const handleAnswer = (answer: string) => {
    if (!sessionId || !currentQuestion) return;
    submitAnswer({
      session_id: sessionId,
      question_id: currentQuestion.id,
      answer,
    });
  };

  const handleReset = () => {
    resetSession();
    reset();
  };

  // Show setup if no active session
  if (!sessionId) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Mock Interview</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Practice with AI-generated questions tailored to your target role and experience level.
          </p>
        </div>

        <div className="mx-auto max-w-lg">
          <div className="rounded-xl border border-border bg-card p-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="rounded-lg bg-primary/10 p-2.5 text-primary">
                <Mic size={20} />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-foreground">Setup Interview</h2>
                <p className="text-xs text-muted-foreground">Configure your mock interview session</p>
              </div>
            </div>
            {error && <ErrorAlert error={error} onRetry={reset} onDismiss={reset} className="mb-4" />}
            <SessionSetup onStart={handleStart} isLoading={isSubmitting} />
          </div>
        </div>
      </div>
    );
  }

  // Show summary if complete
  if (isComplete && summary) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Interview Results</h1>
        </div>
        <div className="mx-auto max-w-2xl">
          <SessionSummaryCard summary={summary} role={role} level={level} onReset={handleReset} />
        </div>
      </div>
    );
  }

  // Active interview
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Mock Interview</h1>
          <p className="text-sm text-muted-foreground mt-1">
            {role} — {level} level
          </p>
        </div>
        <button
          onClick={handleReset}
          className="rounded-lg border border-border px-3 py-1.5 text-sm text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
        >
          End Session
        </button>
      </div>

      {error && <ErrorAlert error={error} onRetry={reset} onDismiss={reset} />}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Question + Answer */}
        <div className="space-y-4">
          {currentQuestion ? (
            <>
              <QuestionCard
                question={currentQuestion}
                current={(answers ?? []).length + 1}
                total={totalQuestions}
              />
              <AnswerInput onSubmit={handleAnswer} isLoading={isSubmitting} />
            </>
          ) : (
            <LoadingSkeleton lines={6} />
          )}
        </div>

        {/* Feedback history */}
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-foreground">
            Previous Answers ({(answers ?? []).length})
          </h3>
          {(answers ?? []).length === 0 && (
            <p className="text-sm text-muted-foreground">No answers submitted yet.</p>
          )}
          {[...(answers ?? [])].reverse().map((record, i) => (
            <FeedbackPanel key={i} feedback={record.feedback} />
          ))}
        </div>
      </div>
    </div>
  );
}
