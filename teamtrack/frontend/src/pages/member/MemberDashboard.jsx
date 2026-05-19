import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { weekApi } from '../../api/services';
import { useAuth } from '../../context/AuthContext';
import { format } from 'date-fns';
import { Plus, CalendarDays, CheckCircle2, Clock, AlertCircle } from 'lucide-react';

const STATUS_BADGE = {
  DRAFT:     'bg-gray-100 text-gray-600',
  SUBMITTED: 'bg-blue-100 text-blue-700',
  APPROVED:  'bg-green-100 text-green-700',
};

function StatCard({ icon: Icon, label, value, color }) {
  return (
    <div className="card flex items-center gap-4">
      <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${color}`}>
        <Icon size={22} />
      </div>
      <div>
        <p className="text-2xl font-bold text-gray-900">{value}</p>
        <p className="text-sm text-gray-500">{label}</p>
      </div>
    </div>
  );
}

export default function MemberDashboard() {
  const { user } = useAuth();
  const { data, isLoading } = useQuery({
    queryKey: ['my-weeks'],
    queryFn: () => weekApi.getMyWeeks().then(r => r.data.data),
  });

  const weeks = data || [];
  const approved = weeks.filter(w => w.status === 'APPROVED').length;
  const submitted = weeks.filter(w => w.status === 'SUBMITTED').length;
  const totalHours = weeks.reduce((sum, w) => sum + (w.totalHours || 0), 0);

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">My Dashboard</h1>
          <p className="text-gray-500 mt-1">Welcome back, {user?.name} 👋</p>
        </div>
        <Link to="/weeks/new" className="btn-primary flex items-center gap-2">
          <Plus size={18} /> New Week
        </Link>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 mb-8">
        <StatCard icon={CalendarDays} label="Total Weeks" value={weeks.length} color="bg-blue-100 text-blue-600" />
        <StatCard icon={CheckCircle2} label="Approved" value={approved} color="bg-green-100 text-green-600" />
        <StatCard icon={AlertCircle} label="Pending Review" value={submitted} color="bg-yellow-100 text-yellow-600" />
        <StatCard icon={Clock} label="Total Hours" value={`${totalHours.toFixed(1)}h`} color="bg-purple-100 text-purple-600" />
      </div>

      {/* Week List */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Weekly Submissions</h2>
        {isLoading ? (
          <div className="text-center py-10 text-gray-400">Loading...</div>
        ) : weeks.length === 0 ? (
          <div className="text-center py-10">
            <CalendarDays size={40} className="mx-auto text-gray-300 mb-3" />
            <p className="text-gray-500">No weeks yet.</p>
            <Link to="/weeks/new" className="btn-primary mt-4 inline-flex items-center gap-2">
              <Plus size={16} /> Create your first week
            </Link>
          </div>
        ) : (
          <div className="space-y-3">
            {weeks.map(week => (
              <Link key={week.id} to={`/weeks/${week.id}`}
                className="flex items-center justify-between p-4 rounded-lg border border-gray-100 hover:border-blue-200 hover:bg-blue-50 transition-colors">
                <div>
                  <p className="font-medium text-gray-900">{week.weekLabel}</p>
                  <p className="text-sm text-gray-400">
                    {week.startDate && format(new Date(week.startDate), 'dd MMM')} –{' '}
                    {week.endDate && format(new Date(week.endDate), 'dd MMM yyyy')}
                  </p>
                </div>
                <div className="flex items-center gap-4 text-sm">
                  <span className="text-gray-500">{week.totalTasks} tasks · {week.totalHours?.toFixed(1)}h</span>
                  <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${STATUS_BADGE[week.status]}`}>
                    {week.status}
                  </span>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
