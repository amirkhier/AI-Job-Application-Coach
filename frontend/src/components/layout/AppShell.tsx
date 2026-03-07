import { Outlet } from 'react-router-dom';
import { cn } from '@/lib/cn';
import { useUIStore } from '@/stores';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { ToastContainer } from '@/components/shared/Toast';

export function AppShell() {
  const sidebarOpen = useUIStore((s) => s.sidebarOpen);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Sidebar />
      <TopBar />
      <main
        className={cn(
          'pt-14 transition-all duration-300',
          sidebarOpen ? 'pl-60' : 'pl-16',
        )}
      >
        <div className="p-6">
          <Outlet />
        </div>
      </main>
      <ToastContainer />
    </div>
  );
}
