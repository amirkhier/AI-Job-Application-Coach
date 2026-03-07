import { useMutation } from '@tanstack/react-query';
import { searchJobs, matchJobs, getLocationInfo } from '@/api/endpoints';
import type { JobSearchRequest } from '@/api/types';
import { useUIStore } from '@/stores';

export function useJobSearch() {
  const addToast = useUIStore((s) => s.addToast);

  const search = useMutation({
    mutationFn: (data: JobSearchRequest) => searchJobs(data),
    onError: () => addToast({ type: 'error', title: 'Job search failed' }),
  });

  const match = useMutation({
    mutationFn: (data: { resume_text: string; query?: string; location?: string }) =>
      matchJobs(data),
    onError: () => addToast({ type: 'error', title: 'Job matching failed' }),
  });

  const location = useMutation({
    mutationFn: (loc: string) => getLocationInfo(loc),
    onError: () => addToast({ type: 'error', title: 'Location lookup failed' }),
  });

  return { search, match, location, error: search.error, reset: search.reset };
}
