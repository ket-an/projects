import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { weekApi } from '../../api/services';
import { format } from 'date-fns';
import { Users, CheckCircle2, Clock, Filter } from 'lucide-react';

const STATUS_STYLES = { DRAFT: 'bg-gray-100 text-gray-600', SUBMITTED: 'bg-blue-100 text-blue-700', APPROVED: 'bg-green-100 text-green-700' };

export default function ManagerDashboard() {
  const [statusFilter, setStatusFilter] = useState('');
  const [page, setPage] = useState(0);

  const { data, isLoading } = useQuery({
    queryKey: ['manager-weeks', statusFilter, page],
    queryFn: () => weekApi.getAllWeeks({ status: statusFilter || undefined, page, size: 10 }).then(r => r.data.data),
  });

  const weeks = data?.content || [];
  const totalPages = data?.totalPages || 0;
  const totalElements = data?.totalElements || 0;

  const submitted = weeks.filter(w => w.status === 'SUBMITTED').length;
  const approved = weeks.filter(w => w.status === 'APPROVED').length;

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Team Dashboard</h1>
        <p className="text-gray-500 mt-1">Review and approve team submissions</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
        <div className="card flex items-center gap-4">
          <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center"><Users size={22} className="text-blue-600" /></div>
          <div><p className="text-2xl font-bold">{totalElements}</p><p className="text-sm text-gray-500">Total Submissions</p></div>
        </div>
        <div className="card flex items-center gap-4">
          <div className="w-12 h-12 bg-yellow-100 rounded-xl flex items-center justify-center"><Clock size={22} className="text-yellow-600" /></div>
          <div><p className="text-2xl font-bold">{submitted}</p><p className="text-sm text-gray-500">Pending Review</p></div>
        </div>
        <div className="card flex items-center gap-4">
          <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center"><CheckCircle2 size={22} className="text-green-600" /></div>
          <div><p className="text-2xl font-bold">{approved}</p><p className="text-sm text-gray-500">Approved</p></div>
        </div>
      </div>

      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Team Submissions</h2>
          <div className="flex items-center gap-2">
            <Filter size={16} className="text-gray-400" />
            <select className="input w-40 text-sm" value={statusFilter} onChange={e => { setStatusFilter(e.target.value); setPage(0); }}>
              <option value="">All Status</option>
              <option value="DRAFT">Draft</option>
              <option value="SUBMITTED">Submitted</option>
              <option value="APPROVED">Approved</option>
            </select>
          </div>
        </div>

        {isLoading ? (
          <p className="text-center py-8 text-gray-400">Loading...</p>
        ) : weeks.length === 0 ? (
          <p className="text-center py-8 text-gray-400">No submissions found</p>
        ) : (
          <>
            <div className="space-y-2">
              {weeks.map(week => (
                <Link key={week.id} to={`/manager/weeks/${week.id}`}
                  className="flex items-center justify-between p-4 rounded-lg border border-gray-100 hover:border-blue-200 hover:bg-blue-50 transition-colors">
                  <div>
                    <p className="font-medium text-gray-900">{week.userName}</p>
                    <p className="text-sm text-gray-500">{week.weekLabel}
                      {week.startDate && ` · ${format(new Date(week.startDate), 'dd MMM')} – ${format(new Date(week.endDate), 'dd MMM yyyy')}`}
                    </p>
                  </div>
                  <div className="flex items-center gap-4 text-sm">
                    <span className="text-gray-400">{week.totalTasks} tasks · {week.totalHours?.toFixed(1)}h</span>
                    <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${STATUS_STYLES[week.status]}`}>{week.status}</span>
                  </div>
                </Link>
              ))}
            </div>

            {totalPages > 1 && (
              <div className="flex justify-center gap-2 mt-4">
                <button disabled={page === 0} onClick={() => setPage(p => p - 1)} className="btn-secondary text-sm px-3">Previous</button>
                <span className="text-sm text-gray-500 self-center">Page {page + 1} of {totalPages}</span>
                <button disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)} className="btn-secondary text-sm px-3">Next</button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
