import { lazy, Suspense } from 'react';
import { Routes, Route } from 'react-router-dom';
import { AppShell } from '@/components/layout/AppShell';
import { ErrorBoundary } from '@/components/shared/ErrorBoundary';
import { PageSkeleton } from '@/components/shared/LoadingSkeleton';
import { ROUTES } from '@/lib/constants';

// Lazy-loaded pages
const DashboardPage = lazy(() => import('@/pages/DashboardPage'));
const ChatPage = lazy(() => import('@/pages/ChatPage'));
const ResumePage = lazy(() => import('@/pages/ResumePage'));
const InterviewPage = lazy(() => import('@/pages/InterviewPage'));
const JobsPage = lazy(() => import('@/pages/JobsPage'));
const KnowledgePage = lazy(() => import('@/pages/KnowledgePage'));
const ApplicationsPage = lazy(() => import('@/pages/ApplicationsPage'));
const ProfilePage = lazy(() => import('@/pages/ProfilePage'));
const SettingsPage = lazy(() => import('@/pages/SettingsPage'));
const NotFoundPage = lazy(() => import('@/pages/NotFoundPage'));

function SuspenseFallback() {
  return (
    <div className="p-6">
      <PageSkeleton />
    </div>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <Suspense fallback={<SuspenseFallback />}>
        <Routes>
          <Route element={<AppShell />}>
            <Route path={ROUTES.DASHBOARD} element={<DashboardPage />} />
            <Route path={ROUTES.CHAT} element={<ChatPage />} />
            <Route path={ROUTES.RESUME} element={<ResumePage />} />
            <Route path={ROUTES.INTERVIEW} element={<InterviewPage />} />
            <Route path={ROUTES.JOBS} element={<JobsPage />} />
            <Route path={ROUTES.KNOWLEDGE} element={<KnowledgePage />} />
            <Route path={ROUTES.APPLICATIONS} element={<ApplicationsPage />} />
            <Route path={ROUTES.PROFILE} element={<ProfilePage />} />
            <Route path={ROUTES.SETTINGS} element={<SettingsPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </Suspense>
    </ErrorBoundary>
  );
}

export default App;
