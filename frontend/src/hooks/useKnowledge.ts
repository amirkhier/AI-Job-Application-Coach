import { useMutation } from '@tanstack/react-query';
import { askQuestion } from '@/api/endpoints';
import type { KnowledgeQueryRequest } from '@/api/types';
import { useUIStore } from '@/stores';

export function useKnowledge() {
  const addToast = useUIStore((s) => s.addToast);

  const ask = useMutation({
    mutationFn: (data: KnowledgeQueryRequest) => askQuestion(data),
    onError: () => addToast({ type: 'error', title: 'Knowledge query failed' }),
  });

  return { ask, error: ask.error, reset: ask.reset };
}
