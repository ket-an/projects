import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { reportApi } from '../../api/services';
import { useAuth } from '../../context/AuthContext';
import toast from 'react-hot-toast';
import { FileBarChart2, Download, Plus } from 'lucide-react';
import { format } from 'date-fns';

export default function ReportsPage() {
  const { user } = useAuth();
  const qc = useQueryClient();
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({ teamId: user?.teamId || 'TEAM-ALPHA', quarter: 'Q1', year: new Date().getFullYear(), format: 'XLSX' });

  const { data: reports = [], isLoading } = useQuery({
    queryKey: ['reports'],
    queryFn: () => reportApi.getAll().then(r => r.data.data),
  });

  const generateMut = useMutation({
    mutationFn: () => reportApi.generate({ ...form, year: parseInt(form.year) }),
    onSuccess: () => { qc.invalidateQueries(['reports']); toast.success('Report generated!'); setShowForm(false); },
    onError: (err) => toast.error(err.response?.data?.error || 'Generation failed'),
  });

  const downloadMut = useMutation({
    mutationFn: (id) => reportApi.getDownloadUrl(id).then(r => r.data.data),
    onSuccess: (url) => window.open(url, '_blank'),
  });

  return (
    <div className="p-8 max-w-3xl">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Quarterly Reports</h1>
          <p className="text-gray-500 mt-1">Generate and download work analysis reports</p>
        </div>
        <button onClick={() => setShowForm(s => !s)} className="btn-primary flex items-center gap-2">
          <Plus size={16} /> Generate Report
        </button>
      </div>

      {showForm && (
        <div className="card mb-6">
          <h2 className="text-lg font-semibold mb-4">New Report</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Team ID</label>
              <input className="input" value={form.teamId} onChange={e => setForm(p => ({ ...p, teamId: e.target.value }))} />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Quarter</label>
              <select className="input" value={form.quarter} onChange={e => setForm(p => ({ ...p, quarter: e.target.value }))}>
                {['Q1', 'Q2', 'Q3', 'Q4'].map(q => <option key={q}>{q}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Year</label>
              <input type="number" className="input" value={form.year} onChange={e => setForm(p => ({ ...p, year: e.target.value }))} min="2020" max="2099" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Format</label>
              <select className="input" value={form.format} onChange={e => setForm(p => ({ ...p, format: e.target.value }))}>
                <option value="XLSX">Excel (XLSX)</option>
                <option value="PDF">PDF</option>
              </select>
            </div>
          </div>
          <div className="flex gap-3 mt-4">
            <button className="btn-primary" onClick={() => generateMut.mutate()} disabled={generateMut.isPending}>
              {generateMut.isPending ? 'Generating…' : 'Generate'}
            </button>
            <button className="btn-secondary" onClick={() => setShowForm(false)}>Cancel</button>
          </div>
        </div>
      )}

      <div className="card">
        <h2 className="text-lg font-semibold mb-4">Generated Reports</h2>
        {isLoading ? (
          <p className="text-center py-6 text-gray-400">Loading…</p>
        ) : reports.length === 0 ? (
          <div className="text-center py-10">
            <FileBarChart2 size={40} className="mx-auto text-gray-300 mb-3" />
            <p className="text-gray-500">No reports yet. Generate your first quarterly report.</p>
          </div>
        ) : (
          <div className="space-y-3">
            {reports.map(r => (
              <div key={r.id} className="flex items-center justify-between p-4 border border-gray-100 rounded-lg">
                <div>
                  <p className="font-medium text-gray-900">{r.quarter} {r.year} — {r.teamId}</p>
                  <p className="text-sm text-gray-400">{r.format} · {r.generatedAt && format(new Date(r.generatedAt), 'dd MMM yyyy HH:mm')}</p>
                </div>
                <button onClick={() => downloadMut.mutate(r.id)} disabled={downloadMut.isPending}
                  className="btn-secondary flex items-center gap-2 text-sm py-1.5">
                  <Download size={14} /> Download
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
