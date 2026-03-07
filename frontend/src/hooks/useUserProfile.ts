import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getUserProfile, getUserContext, getUserInsights, updateUserProfile } from '@/api/endpoints';
import type { ProfileUpdateRequest } from '@/api/types';
import { useAuthStore, useUIStore } from '@/stores';

/** Split a value into an array: if it's already an array return it,
 *  if it's a non-empty string split by sentence-ending punctuation,
 *  otherwise return []. */
function toArray(v: unknown): string[] {
  if (Array.isArray(v)) return v as string[];
  if (typeof v === 'string' && v.trim()) {
    // Split on ". " or newline, trim, drop empties
    return v.split(/(?:\.\s|\n)+/).map((s) => s.trim()).filter(Boolean);
  }
  return [];
}

export function useUserProfile(userId?: number) {
  const qc = useQueryClient();
  const storeUserId = useAuthStore((s) => s.userId);
  const addToast = useUIStore((s) => s.addToast);
  const effectiveId = userId ?? storeUserId;

  const profileQuery = useQuery({
    queryKey: ['user-profile', effectiveId],
    queryFn: () => getUserProfile(effectiveId),
    enabled: !!effectiveId,
  });

  const contextQuery = useQuery({
    queryKey: ['user-context', effectiveId],
    queryFn: () => getUserContext(effectiveId),
    enabled: !!effectiveId,
  });

  const insightsQuery = useQuery({
    queryKey: ['user-insights', effectiveId],
    queryFn: () => getUserInsights(effectiveId),
    enabled: !!effectiveId,
  });

  const updateProfile = useMutation({
    mutationFn: (data: ProfileUpdateRequest) => updateUserProfile(effectiveId, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['user-profile', effectiveId] });
      addToast({ type: 'success', title: 'Profile updated' });
    },
    onError: () => addToast({ type: 'error', title: 'Failed to update profile' }),
  });

  // ── Normalize profile ─────────────────────────────────────────────
  const rawProfile = profileQuery.data as Record<string, unknown> | undefined;
  const pd = (rawProfile?.profile_data ?? rawProfile ?? {}) as Record<string, unknown>;
  const prefs = (rawProfile?.preferences ?? {}) as Record<string, unknown>;
  const profile = rawProfile
    ? {
        id: (rawProfile.id as number) ?? effectiveId,
        skills: Array.isArray(pd.skills) ? (pd.skills as string[]) : [],
        experience_level: (pd.experience_level as string) ?? (pd.experience_years ? `${pd.experience_years} years` : ''),
        experience_years: typeof pd.experience_years === 'number' ? pd.experience_years : (pd.experience_years ? Number(pd.experience_years) : 0),
        target_roles: Array.isArray(pd.target_roles) ? (pd.target_roles as string[]) : Array.isArray(prefs.preferred_roles) ? (prefs.preferred_roles as string[]) : [],
        career_goals: (pd.career_goals as string) ?? '',
        preferences: prefs,
      }
    : null;

  // ── Normalize context ─────────────────────────────────────────────
  // Backend may return context_summary as string or as object
  const rawCtx = contextQuery.data as Record<string, unknown> | undefined;
  const context = rawCtx
    ? {
        user_profile: rawCtx.user_profile ?? null,
        recent_conversations: Array.isArray(rawCtx.recent_conversations) ? rawCtx.recent_conversations : [],
        context_summary: typeof rawCtx.context_summary === 'object' && rawCtx.context_summary !== null
          ? (rawCtx.context_summary as { total_conversations: number; top_intents: string[] })
          : { total_conversations: 0, top_intents: [] as string[], text: rawCtx.context_summary as string ?? '' },
        history_count: (rawCtx.history_count as number) ?? 0,
      }
    : null;

  // ── Normalize insights ────────────────────────────────────────────
  // Backend may return insights/patterns/recommendations as strings or arrays
  const rawIns = insightsQuery.data as Record<string, unknown> | undefined;
  const insights = rawIns
    ? {
        insights: toArray(rawIns.insights),
        patterns: toArray(rawIns.patterns),
        recommendations: toArray(rawIns.recommendations),
        conversation_count: (rawIns.conversation_count as number) ?? 0,
        agent_usage: (rawIns.agent_usage as Record<string, number>) ?? {},
      }
    : null;

  return {
    profile,
    context,
    insights,
    updateProfile,
    isLoading: profileQuery.isLoading,
    error: profileQuery.error,
  };
}
