import axios from 'axios';
import type { AppError } from './types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120_000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ── Request interceptor: attach API key + request ID ────
apiClient.interceptors.request.use((config) => {
  const apiKey = localStorage.getItem('api-key');
  if (apiKey) {
    config.headers['X-API-Key'] = apiKey;
  }
  config.headers['X-Request-ID'] = crypto.randomUUID();
  return config;
});

// ── Response interceptor: normalise errors ──────────────
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (axios.isAxiosError(error)) {
      const status = error.response?.status ?? 0;
      const data = error.response?.data as Record<string, unknown> | undefined;

      const appError: AppError = {
        code: status,
        message:
          (data?.detail as string) ??
          (data?.message as string) ??
          error.message,
        detail: JSON.stringify(data) || undefined,
        requestId: error.config?.headers?.['X-Request-ID'] as string | undefined,
        retryable: status === 429 || status >= 500,
      };

      return Promise.reject(appError);
    }
    return Promise.reject(error);
  },
);
