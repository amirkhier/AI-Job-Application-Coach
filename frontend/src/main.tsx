import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';
import './styles/globals.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 0,
    },
  },
});

// Apply persisted theme on load
const theme = localStorage.getItem('theme') || 'system';
const root = document.documentElement;
root.classList.remove('light', 'dark');
if (theme === 'system') {
  root.classList.add(
    window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light',
  );
} else {
  root.classList.add(theme);
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
