import api from './axios';

// ── Auth ──────────────────────────────────────────────────────────────────
export const authApi = {
  register: (data) => api.post('/auth/register', data),
  login: (data) => api.post('/auth/login', data),
  me: () => api.get('/auth/me'),
  updateProfile: (data) => api.put('/auth/me', data),
  changePassword: (data) => api.put('/auth/me/password', data),
};

// ── Weeks ──────────────────────────────────────────────────────────────────
export const weekApi = {
  create: (data) => api.post('/weeks', data),
  getMyWeeks: () => api.get('/weeks/my'),
  getById: (id) => api.get(`/weeks/${id}`),
  submit: (id, data) => api.put(`/weeks/${id}/submit`, data),
  // Manager
  getAllWeeks: (params) => api.get('/manager/weeks', { params }),
  approve: (id) => api.put(`/manager/weeks/${id}/approve`),
};

// ── Tasks ──────────────────────────────────────────────────────────────────
export const taskApi = {
  create: (data) => api.post('/tasks', data),
  getByWeek: (weekId) => api.get('/tasks', { params: { weekId } }),
  update: (id, data) => api.put(`/tasks/${id}`, data),
  delete: (id) => api.delete(`/tasks/${id}`),
  getAttachmentUploadUrl: (id, fileName) =>
    api.post(`/tasks/${id}/attachment-url`, null, { params: { fileName } }),
  confirmAttachment: (id, s3Key) =>
    api.patch(`/tasks/${id}/attachment-confirm`, null, { params: { s3Key } }),
};

// ── Comments ──────────────────────────────────────────────────────────────
export const commentApi = {
  add: (data) => api.post('/comments', data),
  getByTask: (taskId) => api.get('/comments', { params: { taskId } }),
  resolve: (id) => api.put(`/comments/${id}/resolve`),
};

// ── Reports ───────────────────────────────────────────────────────────────
export const reportApi = {
  generate: (data) => api.post('/manager/reports/generate', data),
  getAll: () => api.get('/manager/reports'),
  getDownloadUrl: (id) => api.get(`/manager/reports/${id}/download`),
};

// ── Users (Manager) ───────────────────────────────────────────────────────
export const userApi = {
  getTeamMembers: (teamId) => api.get('/manager/users', { params: { teamId } }),
  getMember: (id) => api.get(`/manager/users/${id}`),
};
