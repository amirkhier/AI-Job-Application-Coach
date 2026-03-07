import { useEffect, useRef, useCallback, useState } from 'react';
import { getTaskStatus } from './endpoints';
import type { TaskStatusResponse } from './types';
import { TASK_POLL_INTERVAL } from '@/lib/constants';

type PollCallback = (status: TaskStatusResponse) => void;

/**
 * Hook for polling an async Celery task until it completes or fails.
 */
export function useTaskPolling(
  taskId: string | null,
  onComplete: PollCallback,
  onError?: (error: unknown) => void,
) {
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!taskId) return;

    const poll = async () => {
      try {
        const status = await getTaskStatus(taskId);
        if (status.status === 'SUCCESS' || status.status === 'FAILURE') {
          stop();
          onComplete(status);
        }
      } catch (err) {
        stop();
        onError?.(err);
      }
    };

    // Start polling immediately, then at interval
    poll();
    intervalRef.current = setInterval(poll, TASK_POLL_INTERVAL);

    return stop;
  }, [taskId, onComplete, onError, stop]);

  return { stop };
}

/**
 * Imperative task polling that returns a promise.
 */
export function pollTaskUntilDone(taskId: string): Promise<TaskStatusResponse> {
  return new Promise((resolve, reject) => {
    const interval = setInterval(async () => {
      try {
        const status = await getTaskStatus(taskId);
        if (status.status === 'SUCCESS' || status.status === 'FAILURE') {
          clearInterval(interval);
          resolve(status);
        }
      } catch (err) {
        clearInterval(interval);
        reject(err);
      }
    }, TASK_POLL_INTERVAL);
  });
}

/**
 * Hook wrapping an async task submission + polling lifecycle.
 */
export function useAsyncTask<TResult = unknown>() {
  const [taskId, setTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<'idle' | 'submitting' | 'polling' | 'done' | 'error'>('idle');
  const [result, setResult] = useState<TResult | null>(null);
  const [error, setError] = useState<unknown>(null);

  const submit = useCallback(async (submitFn: () => Promise<{ task_id: string }>) => {
    setStatus('submitting');
    setResult(null);
    setError(null);
    try {
      const { task_id } = await submitFn();
      setTaskId(task_id);
      setStatus('polling');
      const final = await pollTaskUntilDone(task_id);
      if (final.status === 'SUCCESS') {
        setResult(final.result as TResult);
        setStatus('done');
      } else {
        setError(final.result);
        setStatus('error');
      }
    } catch (err) {
      setError(err);
      setStatus('error');
    }
  }, []);

  const reset = useCallback(() => {
    setTaskId(null);
    setStatus('idle');
    setResult(null);
    setError(null);
  }, []);

  return { taskId, status, result, error, submit, reset };
}
