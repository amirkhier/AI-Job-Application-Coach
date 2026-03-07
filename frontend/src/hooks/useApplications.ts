import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  listApplications,
  createApplication,
  updateApplication,
  deleteApplication,
  getFollowUps,
  batchUpdateApplications,
} from '@/api/endpoints';
import type { ApplicationCreateRequest, ApplicationUpdateRequest } from '@/api/types';
import { useUIStore, useAuthStore } from '@/stores';

const QUERY_KEY = ['applications'];

export function useApplications(userId?: number) {
  const qc = useQueryClient();
  const addToast = useUIStore((s) => s.addToast);
  const storeUserId = useAuthStore((s) => s.userId);
  const effectiveId = userId ?? storeUserId;

  const list = useQuery({
    queryKey: [...QUERY_KEY, effectiveId],
    queryFn: () => listApplications({ user_id: effectiveId }),
  });

  const followUpsQuery = useQuery({
    queryKey: [...QUERY_KEY, 'follow-ups', effectiveId],
    queryFn: () => getFollowUps(effectiveId),
    enabled: !!effectiveId,
  });

  const create = useMutation({
    mutationFn: (data: ApplicationCreateRequest) =>
      createApplication({ ...data, user_id: effectiveId }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: QUERY_KEY });
      addToast({ type: 'success', title: 'Application created' });
    },
    onError: () => addToast({ type: 'error', title: 'Failed to create application' }),
  });

  const update = useMutation({
    mutationFn: ({ id, data }: { id: number; data: ApplicationUpdateRequest }) =>
      updateApplication(id, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: QUERY_KEY });
      addToast({ type: 'success', title: 'Application updated' });
    },
    onError: () => addToast({ type: 'error', title: 'Failed to update application' }),
  });

  const remove = useMutation({
    mutationFn: (id: number) => deleteApplication(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: QUERY_KEY });
      addToast({ type: 'success', title: 'Application deleted' });
    },
    onError: () => addToast({ type: 'error', title: 'Failed to delete application' }),
  });

  const batchUpdate = useMutation({
    mutationFn: (updates: { ids: number[]; status: string }) =>
      batchUpdateApplications(updates),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: QUERY_KEY });
      addToast({ type: 'success', title: 'Batch update complete' });
    },
    onError: () => addToast({ type: 'error', title: 'Batch update failed' }),
  });

  // Normalize follow-ups: backend returns { follow_ups: [...], count: N }
  const rawFollowUps = followUpsQuery.data;
  const followUps = Array.isArray(rawFollowUps)
    ? rawFollowUps
    : (rawFollowUps as Record<string, unknown>)?.follow_ups ?? [];

  return {
    applications: Array.isArray(list.data) ? list.data : [],
    followUps: followUps as typeof list.data,
    isLoading: list.isLoading,
    error: list.error,
    create,
    update,
    remove,
    batchUpdate,
  };
}
