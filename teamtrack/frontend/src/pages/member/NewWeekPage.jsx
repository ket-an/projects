import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { weekApi } from '../../api/services';
import toast from 'react-hot-toast';
import { format, startOfWeek, endOfWeek } from 'date-fns';

export default function NewWeekPage() {
  const navigate = useNavigate();
  const today = new Date();
  const [form, setForm] = useState({
    weekLabel: `Week ${format(today, 'w, yyyy')}`,
    startDate: format(startOfWeek(today, { weekStartsOn: 1 }), 'yyyy-MM-dd'),
    endDate: format(endOfWeek(today, { weekStartsOn: 1 }), 'yyyy-MM-dd'),
  });
  const [loading, setLoading] = useState(false);

  const set = (field) => (e) => setForm(p => ({ ...p, [field]: e.target.value }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await weekApi.create(form);
      toast.success('Week created!');
      navigate(`/weeks/${res.data.data.id}`);
    } catch (err) {
      toast.error(err.response?.data?.error || 'Failed to create week');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 max-w-xl">
      <h1 className="text-2xl font-bold text-gray-900 mb-2">Create New Week</h1>
      <p className="text-gray-500 mb-8">Set up your weekly task tracker</p>

      <div className="card">
        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Week Label</label>
            <input className="input" value={form.weekLabel} onChange={set('weekLabel')} required
              placeholder="e.g. Week 21, 2026" />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Start Date</label>
              <input type="date" className="input" value={form.startDate} onChange={set('startDate')} required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">End Date</label>
              <input type="date" className="input" value={form.endDate} onChange={set('endDate')} required />
            </div>
          </div>
          <div className="flex gap-3 pt-2">
            <button type="submit" disabled={loading} className="btn-primary">
              {loading ? 'Creating…' : 'Create Week'}
            </button>
            <button type="button" className="btn-secondary" onClick={() => navigate('/dashboard')}>
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
