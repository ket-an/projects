import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { weekApi, taskApi, commentApi } from '../../api/services';
import toast from 'react-hot-toast';
import { CheckCircle, MessageSquare, Clock, AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react';

const STATUS_COLORS = { TODO: 'bg-gray-100 text-gray-600', IN_PROGRESS: 'bg-yellow-100 text-yellow-700', COMPLETED: 'bg-green-100 text-green-700', BLOCKED: 'bg-red-100 text-red-700' };
const COMMENT_TYPES = ['APPROVAL', 'DOUBT', 'FEEDBACK'];
const COMMENT_STYLES = { APPROVAL: 'bg-green-50 border-green-200', DOUBT: 'bg-yellow-50 border-yellow-200', FEEDBACK: 'bg-blue-50 border-blue-200' };

function AddCommentForm({ taskId, onAdded }) {
  const [form, setForm] = useState({ body: '', type: 'FEEDBACK' });
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      await commentApi.add({ ...form, taskId });
      toast.success('Comment added');
      setForm({ body: '', type: 'FEEDBACK' });
      onAdded();
    } catch (err) { toast.error('Failed to add comment'); }
    finally { setLoading(false); }
  };

  return (
    <form onSubmit={submit} className="mt-3 space-y-2">
      <select className="input text-sm" value={form.type} onChange={e => setForm(p => ({ ...p, type: e.target.value }))}>
        {COMMENT_TYPES.map(t => <option key={t}>{t}</option>)}
      </select>
      <textarea className="input min-h-[70px] text-sm" placeholder="Write your comment..." value={form.body} required
        onChange={e => setForm(p => ({ ...p, body: e.target.value }))} />
      <button type="submit" disabled={loading} className="btn-primary text-sm py-1.5">
        {loading ? 'Adding…' : 'Add Comment'}
      </button>
    </form>
  );
}

function TaskReviewCard({ task }) {
  const qc = useQueryClient();
  const [expanded, setExpanded] = useState(false);
  const [showCommentForm, setShowCommentForm] = useState(false);

  const { data: comments = [] } = useQuery({
    queryKey: ['comments', task.id],
    queryFn: () => commentApi.getByTask(task.id).then(r => r.data.data),
    enabled: expanded,
  });

  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden">
      <div className="p-4 cursor-pointer" onClick={() => setExpanded(s => !s)}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 flex-1">
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${STATUS_COLORS[task.status]}`}>{task.status}</span>
            <h4 className="font-medium text-gray-900">{task.title}</h4>
            {task.blocker && <AlertTriangle size={14} className="text-red-500" />}
            {task.unresolvedComments > 0 && (
              <span className="text-xs bg-orange-100 text-orange-700 px-2 py-0.5 rounded-full">{task.unresolvedComments} pending</span>
            )}
          </div>
          <div className="flex items-center gap-3 text-sm text-gray-400">
            <span className="flex items-center gap-1"><Clock size={12} />{task.hoursSpent}h</span>
            {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </div>
        </div>
      </div>

      {expanded && (
        <div className="px-4 pb-4 border-t border-gray-100 pt-3 space-y-3">
          <p className="text-sm text-gray-700">{task.description}</p>

          {task.blocker && (
            <div className="p-2 bg-red-50 rounded text-sm text-red-700">
              <strong>Blocker:</strong> {task.blocker}
            </div>
          )}

          {task.evidenceLinks?.length > 0 && (
            <div>
              <p className="text-xs text-gray-400 mb-1">Evidence:</p>
              {task.evidenceLinks.map((l, i) => (
                <a key={i} href={l} target="_blank" rel="noreferrer" className="block text-blue-600 text-xs hover:underline">{l}</a>
              ))}
            </div>
          )}

          {/* Existing comments */}
          {comments.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-gray-400 font-medium uppercase tracking-wide">Comments</p>
              {comments.map(c => (
                <div key={c.id} className={`p-3 rounded-lg border text-sm ${COMMENT_STYLES[c.type]} ${c.resolved ? 'opacity-60' : ''}`}>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium text-gray-800">{c.authorName}</span>
                    <span className="text-xs px-1.5 py-0.5 rounded bg-white border">{c.type}</span>
                    {c.resolved && <span className="text-xs text-green-600">✓ Resolved</span>}
                  </div>
                  <p className="text-gray-700">{c.body}</p>
                </div>
              ))}
            </div>
          )}

          <div>
            <button onClick={() => setShowCommentForm(s => !s)}
              className="text-sm text-blue-600 hover:underline flex items-center gap-1">
              <MessageSquare size={14} /> {showCommentForm ? 'Cancel' : 'Add Comment'}
            </button>
            {showCommentForm && (
              <AddCommentForm taskId={task.id} onAdded={() => { qc.invalidateQueries(['comments', task.id]); setShowCommentForm(false); }} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default function ManagerWeekDetail() {
  const { weekId } = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();

  const { data: week, isLoading: wl } = useQuery({ queryKey: ['week', weekId], queryFn: () => weekApi.getById(weekId).then(r => r.data.data) });
  const { data: tasks = [], isLoading: tl } = useQuery({ queryKey: ['tasks', weekId], queryFn: () => taskApi.getByWeek(weekId).then(r => r.data.data) });

  const approveMut = useMutation({
    mutationFn: () => weekApi.approve(weekId),
    onSuccess: () => { qc.invalidateQueries(['week', weekId]); qc.invalidateQueries(['manager-weeks']); toast.success('Week approved!'); }
  });

  if (wl || tl) return <div className="p-8 text-gray-400">Loading…</div>;

  const totalHours = tasks.reduce((s, t) => s + t.hoursSpent, 0);
  const completed = tasks.filter(t => t.status === 'COMPLETED').length;
  const blocked = tasks.filter(t => t.status === 'BLOCKED').length;

  const STATUS_STYLES = { DRAFT: 'bg-gray-100 text-gray-600', SUBMITTED: 'bg-blue-100 text-blue-700', APPROVED: 'bg-green-100 text-green-700' };

  return (
    <div className="p-8 max-w-3xl">
      <button onClick={() => navigate('/manager')} className="text-sm text-gray-400 hover:text-gray-600 mb-4">← Back to Team Dashboard</button>

      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{week?.weekLabel}</h1>
          <p className="text-gray-500 mt-1">Submitted by <strong>{week?.userName}</strong></p>
          <div className="flex items-center gap-3 mt-2">
            <span className={`px-3 py-1 rounded-full text-xs font-medium ${STATUS_STYLES[week?.status]}`}>{week?.status}</span>
          </div>
        </div>
        {week?.status === 'SUBMITTED' && (
          <button className="btn-primary flex items-center gap-2" onClick={() => approveMut.mutate()} disabled={approveMut.isPending}>
            <CheckCircle size={16} /> {approveMut.isPending ? 'Approving…' : 'Approve Week'}
          </button>
        )}
        {week?.status === 'APPROVED' && (
          <div className="text-green-600 font-medium flex items-center gap-2"><CheckCircle size={18} /> Approved</div>
        )}
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        {[
          { label: 'Total Tasks', value: tasks.length },
          { label: 'Completed', value: completed },
          { label: 'Blocked', value: blocked },
          { label: 'Total Hours', value: `${totalHours.toFixed(1)}h` },
        ].map(s => (
          <div key={s.label} className="card py-3 text-center">
            <p className="text-xl font-bold text-gray-900">{s.value}</p>
            <p className="text-xs text-gray-500">{s.label}</p>
          </div>
        ))}
      </div>

      {/* Submission note */}
      {week?.submissionNote && (
        <div className="mb-4 p-4 bg-blue-50 rounded-xl border border-blue-100 text-sm text-gray-700">
          <strong>Note from member:</strong> {week.submissionNote}
        </div>
      )}

      {/* Task review cards */}
      <div className="space-y-3">
        <h2 className="text-lg font-semibold text-gray-900">Tasks ({tasks.length})</h2>
        {tasks.length === 0 ? (
          <p className="text-gray-400 text-center py-6">No tasks in this week</p>
        ) : tasks.map(t => <TaskReviewCard key={t.id} task={t} />)}
      </div>
    </div>
  );
}
