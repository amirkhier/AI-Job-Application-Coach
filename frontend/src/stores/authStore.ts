import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface AuthState {
  apiKey: string | null;
  userId: number;
  setApiKey: (key: string | null) => void;
  setUserId: (id: number) => void;
  clearAuth: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      apiKey: null,
      userId: 1,
      setApiKey: (key) => {
        set({ apiKey: key });
        if (key) {
          localStorage.setItem('api-key', key);
        } else {
          localStorage.removeItem('api-key');
        }
      },
      setUserId: (id) => set({ userId: id }),
      clearAuth: () => {
        localStorage.removeItem('api-key');
        set({ apiKey: null, userId: 1 });
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ apiKey: state.apiKey, userId: state.userId }),
    },
  ),
);
