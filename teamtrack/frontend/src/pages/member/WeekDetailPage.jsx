import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { weekApi, taskApi, commentApi } from '../../api/services';
import toast from 'react-hot-toast';
import { Plus, Trash2, MessageSquare, ChevronDown, ChevronUp, CheckCircle, Clock, AlertTriangle, Send } from 'lucide-react';

const TASK_STATUSES = ['TODO', 'IN_PROGRESS', 'COMPLETED', 'BLOCKED'];
const STATUS_COLORS = { TODO: 'bg-gray-100 text-gray-600', IN_PROGRESS: 'bg-yellow-100 text-yellow-700', COMPLETED: 'bg-green-100 text-green-700', BLOCKED: 'bg-red-100 text-red-700' };

function TaskForm({ weekId, onCreated }) {
  const [form, setForm] = useState({ title: '', description: '', status: 'TODO', hoursSpent: 0, blocker: '', evidenceLinks: [] });
  const [linkInput, setLinkInput] = useState('');
  const [loading, setLoading] = useState(false);

  const addLink = () => {
    if (linkInput.trim()) { setForm(p => ({ ...p, evidenceLinks: [...p.evidenceLinks, linkInput.trim()] })); setLinkInput(''); }
  };

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      await taskApi.create({ ...form, weekId });
      toast.success('Task added!');
      setForm({ title: '', description: '', status: 'TODO', hoursSpent: 0, blocker: '', evidenceLinks: [] });
      onCreated();
    } catch (err) { toast.error(err.response?.data?.error || 'Failed'); }
    finally { setLoading(false); }
  };

  return (
    <form onSubmit={submit} className="space-y-3 p-4 bg-blue-50 rounded-xl border border-blue-100">
      <h3 className="font-semibold text-gray-800">Add New Task</h3>
      <input className="input" placeholder="Task title *" value={form.title} onChange={e => setForm(p => ({ ...p, title: e.target.value }))} required />
      <textarea className="input min-h-[80px]" placeholder="Description *" value={form.description} onChange={e => setForm(p => ({ ...p, description: e.target.value }))} required />
      <div className="grid grid-cols-3 gap-3">
        <select className="input" value={form.status} onChange={e => setForm(p => ({ ...p, status: e.target.value }))}>
          {TASK_STATUSES.map(s => <option key={s}>{s}</option>)}
        </select>
        <input type="number" className="input" placeholder="Hours" min="0" max="24" step="0.5"
          value={form.hoursSpent} onChange={e => setForm(p => ({ ...p, hoursSpent: parseFloat(e.target.value) || 0 }))} />
        <input className="input" placeholder="Blocker (optional)" value={form.blocker} onChange={e => setForm(p => ({ ...p, blocker: e.target.value }))} />
      </div>
      <div className="flex gap-2">
        <input className="input flex-1" placeholder="Add evidence link" value={linkInput} onChange={e => setLinkInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && (e.preventDefault(), addLink())} />
        <button type="button" onClick={addLink} className="btn-secondary px-3">+ Link</button>
      </div>
      {form.evidenceLinks.length > 0 && (
        <div className="flex flex-wrap gap-2">{form.evidenceLinks.map((l, i) => (
          <span key={i} className="bg-white text-xs px-2 py-1 rounded border text-blue-600 truncate max-w-[200px]">{l}</span>
        ))}</div>
      )}
      <button type="submit" disabled={loading} className="btn-primary">{loading ? 'Adding…' : 'Add Task'}</button>
    </form>
  );
}

function CommentThread({ taskId }) {
  const qc = useQueryClient();
  const { data: comments = [] } = useQuery({ queryKey: ['comments', taskId], queryFn: () => commentApi.getByTask(taskId).then(r => r.data.data) });
  const resolveMut = useMutation({ mutationFn: (id) => commentApi.resolve(id), onSuccess: () => { qc.invalidateQueries(['comments', taskId]); toast.success('Comment resolved'); } });

  return (
    <div className="mt-3 space-y-2">
      {comments.map(c => (
        <div key={c.id} className={`p-3 rounded-lg border text-sm ${c.resolved ? 'bg-gray-50 opacity-60' : c.type === 'APPROVAL' ? 'bg-green-50 border-green-200' : c.type === 'DOUBT' ? 'bg-yellow-50 border-yellow-200' : 'bg-blue-50 border-blue-200'}`}>
          <div className="flex justify-between items-start">
            <div>
              <span className="font-medium">{c.authorName}</span>
              <span className={`ml-2 text-xs px-1.5 py-0.5 rounded ${c.type === 'APPROVAL' ? 'bg-green-100 text-green-700' : c.type === 'DOUBT' ? 'bg-yellow-100 text-yellow-700' : 'bg-blue-100 text-blue-700'}`}>{c.type}</span>
            </div>
            {!c.resolved && (
              <button onClick={() => resolveMut.mutate(c.id)} className="text-xs text-gray-500 hover:text-green-600 flex items-center gap-1">
                <CheckCircle size={12} /> Resolve
              </button>
            )}
          </div>
          <p className="mt-1 text-gray-700">{c.body}</p>
          {c.resolved && <p className="text-xs text-gray-400 mt-1">✓ Resolved</p>}
        </div>
      ))}
    </div>
  );
}

function TaskCard({ task, weekStatus, onDelete }) {
  const [expanded, setExpanded] = useState(false);
  const [showComments, setShowComments] = useState(false);

  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden">
      <div className="p-4">
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${STATUS_COLORS[task.status]}`}>{task.status}</span>
              {task.blocker && <span className="text-xs bg-red-50 text-red-600 px-2 py-0.5 rounded-full flex items-center gap-1"><AlertTriangle size={10} /> Blocked</span>}
              {task.unresolvedComments > 0 && <span className="text-xs bg-orange-50 text-orange-600 px-2 py-0.5 rounded-full">{task.unresolvedComments} comment{task.unresolvedComments > 1 ? 's' : ''}</span>}
            </div>
            <h4 className="font-medium text-gray-900">{task.title}</h4>
            <div className="flex items-center gap-4 mt-1 text-xs text-gray-400">
              <span className="flex items-center gap-1"><Clock size={11} /> {task.hoursSpent}h</span>
              {task.evidenceLinks?.length > 0 && <span>{task.evidenceLinks.length} link(s)</span>}
            </div>
          </div>
          <div className="flex items-center gap-1">
            <button onClick={() => setShowComments(s => !s)} className="p-1.5 text-gray-400 hover:text-blue-600 rounded">
              <MessageSquare size={16} />
            </button>
            <button onClick={() => setExpanded(s => !s)} className="p-1.5 text-gray-400 hover:text-gray-700 rounded">
              {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>
            {weekStatus === 'DRAFT' && (
              <button onClick={() => onDelete(task.id)} className="p-1.5 text-gray-400 hover:text-red-600 rounded">
                <Trash2 size={16} />
              </button>
            )}
          </div>
        </div>

        {expanded && (
          <div className="mt-3 space-y-2 text-sm text-gray-600">
            <p>{task.description}</p>
            {task.blocker && <p className="text-red-600 bg-red-50 p-2 rounded"><strong>Blocker:</strong> {task.blocker}</p>}
            {task.evidenceLinks?.length > 0 && (
              <div><p className="text-xs text-gray-400 mb-1">Evidence links:</p>
                {task.evidenceLinks.map((l, i) => <a key={i} href={l} target="_blank" rel="noreferrer" className="block text-blue-600 hover:underline text-xs truncate">{l}</a>)}
              </div>
            )}
          </div>
        )}

        {showComments && <CommentThread taskId={task.id} />}
      </div>
    </div>
  );
}

export default function WeekDetailPage() {
  const { weekId } = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [showForm, setShowForm] = useState(false);

  const { data: week, isLoading: weekLoading } = useQuery({ queryKey: ['week', weekId], queryFn: () => weekApi.getById(weekId).then(r => r.data.data) });
  const { data: tasks = [], isLoading: tasksLoading } = useQuery({ queryKey: ['tasks', weekId], queryFn: () => taskApi.getByWeek(weekId).then(r => r.data.data) });

  const deleteMut = useMutation({ mutationFn: taskApi.delete, onSuccess: () => { qc.invalidateQueries(['tasks', weekId]); toast.success('Task deleted'); } });
  const submitMut = useMutation({
    mutationFn: () => weekApi.submit(weekId, {}),
    onSuccess: () => { qc.invalidateQueries(['week', weekId]); qc.invalidateQueries(['my-weeks']); toast.success('Week submitted for review!'); }
  });

  if (weekLoading) return <div className="p-8 text-gray-400">Loading…</div>;
  if (!week) return <div className="p-8 text-gray-400">Week not found</div>;

  const STATUS_BADGE_STYLE = { DRAFT: 'bg-gray-100 text-gray-600', SUBMITTED: 'bg-blue-100 text-blue-700', APPROVED: 'bg-green-100 text-green-700' };

  return (
    <div className="p-8 max-w-3xl">
      <button onClick={() => navigate('/dashboard')} className="text-sm text-gray-400 hover:text-gray-600 mb-4">← Back to Dashboard</button>

      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{week.weekLabel}</h1>
          <p className="text-gray-400 text-sm mt-1">{week.startDate} to {week.endDate}</p>
          <div className="flex items-center gap-3 mt-2">
            <span className={`px-3 py-1 rounded-full text-xs font-medium ${STATUS_BADGE_STYLE[week.status]}`}>{week.status}</span>
            <span className="text-sm text-gray-500">{tasks.length} tasks · {week.totalHours?.toFixed(1)}h total</span>
          </div>
        </div>
        {week.status === 'DRAFT' && (
          <button className="btn-primary flex items-center gap-2" disabled={submitMut.isPending || tasks.length === 0}
            onClick={() => submitMut.mutate()}>
            <Send size={16} /> {submitMut.isPending ? 'Submitting…' : 'Submit for Review'}
          </button>
        )}
        {week.status === 'APPROVED' && (
          <div className="text-green-600 font-medium flex items-center gap-2"><CheckCircle size={18} /> Approved by {week.approvedBy}</div>
        )}
      </div>

      {/* Tasks */}
      <div className="space-y-3 mb-4">
        {tasksLoading ? <p className="text-gray-400">Loading tasks…</p> : tasks.length === 0 ? (
          <div className="text-center py-10 text-gray-400">No tasks yet. Add one below.</div>
        ) : tasks.map(t => (
          <TaskCard key={t.id} task={t} weekStatus={week.status} onDelete={(id) => deleteMut.mutate(id)} />
        ))}
      </div>

      {week.status === 'DRAFT' && (
        <>
          <button onClick={() => setShowForm(s => !s)} className="btn-secondary flex items-center gap-2 mb-3">
            <Plus size={16} /> {showForm ? 'Cancel' : 'Add Task'}
          </button>
          {showForm && <TaskForm weekId={weekId} onCreated={() => { qc.invalidateQueries(['tasks', weekId]); setShowForm(false); }} />}
        </>
      )}
    </div>
  );
}
