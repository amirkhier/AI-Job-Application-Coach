import { useMutation } from '@tanstack/react-query';
import { analyzeResume, improveResume, auditResume } from '@/api/endpoints';
import type { ResumeRequest } from '@/api/types';
import { useUIStore } from '@/stores';

export function useResumeAnalysis() {
  const addToast = useUIStore((s) => s.addToast);

  const analyze = useMutation({
    mutationFn: (data: ResumeRequest) => analyzeResume(data),
    onError: () => addToast({ type: 'error', title: 'Resume analysis failed' }),
  });

  const improve = useMutation({
    mutationFn: (data: ResumeRequest) => improveResume(data),
    onError: () => addToast({ type: 'error', title: 'Resume improvement failed' }),
  });

  const audit = useMutation({
    mutationFn: (data: ResumeRequest) => auditResume(data),
    onSuccess: () => addToast({ type: 'info', title: 'Audit submitted', description: 'Processing in background...' }),
    onError: () => addToast({ type: 'error', title: 'Resume audit failed' }),
  });

  return { analyze, improve, audit };
}
