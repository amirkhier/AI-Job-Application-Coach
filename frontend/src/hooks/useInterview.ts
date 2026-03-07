import { useMutation } from '@tanstack/react-query';
import { startInterview, submitAnswer, getQuestions, generateReport } from '@/api/endpoints';
import { useInterviewStore, useUIStore } from '@/stores';
import type { InterviewStartRequest, InterviewAnswerRequest } from '@/api/types';

export function useInterview() {
  const store = useInterviewStore();
  const addToast = useUIStore((s) => s.addToast);

  const startInterviewMut = useMutation({
    mutationFn: (data: InterviewStartRequest) => startInterview(data),
    onSuccess: (res) => {
      store.startSession(
        res.session_id,
        res.role,
        res.level,
        res.first_question,
        res.total_questions,
      );
      addToast({ type: 'success', title: 'Interview started' });
    },
    onError: () => addToast({ type: 'error', title: 'Failed to start interview' }),
  });

  const submitAnswerMut = useMutation({
    mutationFn: (data: InterviewAnswerRequest) => {
      store.setSubmitting(true);
      return submitAnswer(data);
    },
    onSuccess: (res) => {
      store.setSubmitting(false);
      store.recordAnswer({
        questionId: res.feedback ? store.currentQuestion?.id ?? '' : '',
        answer: '',
        feedback: res.feedback,
      });
      if (res.session_complete && res.session_summary) {
        store.completeSession(res.session_summary);
        addToast({ type: 'success', title: 'Interview complete!' });
      } else if (res.next_question) {
        store.setCurrentQuestion(res.next_question);
      }
    },
    onError: () => {
      store.setSubmitting(false);
      addToast({ type: 'error', title: 'Failed to submit answer' });
    },
  });

  const report = useMutation({
    mutationFn: (sessionId: string) => generateReport(sessionId),
    onError: () => addToast({ type: 'error', title: 'Failed to generate report' }),
  });

  const startInterviewFn = (data: InterviewStartRequest) => startInterviewMut.mutate(data);
  const submitAnswerFn = (data: InterviewAnswerRequest) => submitAnswerMut.mutate(data);

  return {
    startInterview: startInterviewFn,
    submitAnswer: submitAnswerFn,
    report,
    error: startInterviewMut.error || submitAnswerMut.error,
    reset: () => { startInterviewMut.reset(); submitAnswerMut.reset(); },
    ...store,
  };
}
