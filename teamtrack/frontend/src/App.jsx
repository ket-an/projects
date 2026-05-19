import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider, useAuth } from './context/AuthContext';
import LoginPage from './pages/auth/LoginPage';
import RegisterPage from './pages/auth/RegisterPage';
import MemberDashboard from './pages/member/MemberDashboard';
import WeekDetailPage from './pages/member/WeekDetailPage';
import NewWeekPage from './pages/member/NewWeekPage';
import ManagerDashboard from './pages/manager/ManagerDashboard';
import ManagerWeekDetail from './pages/manager/ManagerWeekDetail';
import ReportsPage from './pages/manager/ReportsPage';
import Layout from './components/layout/Layout';

const queryClient = new QueryClient({ defaultOptions: { queries: { retry: 1, staleTime: 30000 } } });

function ProtectedRoute({ children, requireManager = false }) {
  const { user, loading } = useAuth();
  if (loading) return <div style={{display:'flex',alignItems:'center',justifyContent:'center',height:'100vh',color:'#6b7280'}}>Loading...</div>;
  if (!user) return <Navigate to="/login" replace />;
  if (requireManager && user.role !== 'MANAGER') return <Navigate to="/dashboard" replace />;
  return children;
}

function AppRoutes() {
  const { user } = useAuth();
  return (
    <Routes>
      <Route path="/login" element={!user ? <LoginPage /> : <Navigate to={user.role === 'MANAGER' ? '/manager' : '/dashboard'} replace />} />
      <Route path="/register" element={!user ? <RegisterPage /> : <Navigate to="/dashboard" replace />} />
      <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<MemberDashboard />} />
        <Route path="weeks/new" element={<NewWeekPage />} />
        <Route path="weeks/:weekId" element={<WeekDetailPage />} />
      </Route>
      <Route path="/manager" element={<ProtectedRoute requireManager><Layout /></ProtectedRoute>}>
        <Route index element={<ManagerDashboard />} />
        <Route path="weeks/:weekId" element={<ManagerWeekDetail />} />
        <Route path="reports" element={<ReportsPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter>
          <AppRoutes />
          <Toaster position="top-right" toastOptions={{ duration: 3000 }} />
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}
