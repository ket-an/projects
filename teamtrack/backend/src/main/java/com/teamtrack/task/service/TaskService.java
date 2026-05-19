package com.teamtrack.task.service;

import com.teamtrack.auth.model.User;
import com.teamtrack.auth.repository.UserRepository;
import com.teamtrack.comment.repository.CommentRepository;
import com.teamtrack.exception.*;
import com.teamtrack.storage.S3StorageService;
import com.teamtrack.task.dto.TaskDto.*;
import com.teamtrack.task.model.Task;
import com.teamtrack.task.model.TaskStatus;
import com.teamtrack.task.repository.TaskRepository;
import com.teamtrack.util.Role;
import com.teamtrack.week.model.Week;
import com.teamtrack.week.model.WeekStatus;
import com.teamtrack.week.repository.WeekRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class TaskService {

    private final TaskRepository taskRepository;
    private final WeekRepository weekRepository;
    private final UserRepository userRepository;
    private final CommentRepository commentRepository;
    private final S3StorageService s3StorageService;

    @Transactional
    public Response createTask(String userEmail, CreateRequest request) {
        User user = getUserByEmail(userEmail);
        Week week = getWeekOrThrow(request.getWeekId());

        if (!week.getUserId().equals(user.getId())) {
            throw new ForbiddenException("You can only add tasks to your own weeks");
        }
        if (week.getStatus() != WeekStatus.DRAFT) {
            throw new BadRequestException("Cannot add tasks to a submitted or approved week");
        }

        Task task = Task.builder()
            .weekId(week.getId())
            .userId(user.getId())
            .title(request.getTitle())
            .description(request.getDescription())
            .status(request.getStatus() != null ? request.getStatus() : TaskStatus.TODO)
            .hoursSpent(request.getHoursSpent())
            .blocker(request.getBlocker())
            .evidenceLinks(request.getEvidenceLinks())
            .priority(request.getPriority())
            .build();

        if (task.getStatus() == TaskStatus.COMPLETED) {
            task.setCompletedAt(LocalDateTime.now());
        }

        return toResponse(taskRepository.save(task));
    }

    @Transactional(readOnly = true)
    public List<Response> getTasksByWeek(String weekId, String userEmail) {
        User user = getUserByEmail(userEmail);
        Week week = getWeekOrThrow(weekId);

        if (user.getRole() == Role.TEAM_MEMBER && !week.getUserId().equals(user.getId())) {
            throw new ForbiddenException("You can only view your own tasks");
        }

        return taskRepository.findByWeekId(weekId)
            .stream().map(this::toResponse).collect(Collectors.toList());
    }

    @Transactional
    public Response updateTask(String taskId, String userEmail, UpdateRequest request) {
        User user = getUserByEmail(userEmail);
        Task task = getTaskOrThrow(taskId);

        if (!task.getUserId().equals(user.getId())) {
            throw new ForbiddenException("You can only update your own tasks");
        }

        Week week = getWeekOrThrow(task.getWeekId());
        if (week.getStatus() != WeekStatus.DRAFT) {
            throw new BadRequestException("Cannot edit tasks in a submitted or approved week");
        }

        if (request.getTitle() != null) task.setTitle(request.getTitle());
        if (request.getDescription() != null) task.setDescription(request.getDescription());
        if (request.getBlocker() != null) task.setBlocker(request.getBlocker());
        if (request.getEvidenceLinks() != null) task.setEvidenceLinks(request.getEvidenceLinks());
        if (request.getPriority() > 0) task.setPriority(request.getPriority());

        if (request.getStatus() != null && request.getStatus() != task.getStatus()) {
            task.setStatus(request.getStatus());
            if (request.getStatus() == TaskStatus.COMPLETED && task.getCompletedAt() == null) {
                task.setCompletedAt(LocalDateTime.now());
            }
        }
        if (request.getHoursSpent() >= 0) task.setHoursSpent(request.getHoursSpent());

        return toResponse(taskRepository.save(task));
    }

    @Transactional
    public void deleteTask(String taskId, String userEmail) {
        User user = getUserByEmail(userEmail);
        Task task = getTaskOrThrow(taskId);

        if (!task.getUserId().equals(user.getId())) {
            throw new ForbiddenException("You can only delete your own tasks");
        }
        Week week = getWeekOrThrow(task.getWeekId());
        if (week.getStatus() != WeekStatus.DRAFT) {
            throw new BadRequestException("Cannot delete tasks from a submitted week");
        }

        commentRepository.deleteByTaskId(taskId);
        taskRepository.delete(task);
        log.info("Task {} deleted by {}", taskId, userEmail);
    }

    public AttachmentUrlResponse generateAttachmentUploadUrl(String taskId,
                                                               String fileName,
                                                               String userEmail) {
        User user = getUserByEmail(userEmail);
        Task task = getTaskOrThrow(taskId);

        if (!task.getUserId().equals(user.getId())) {
            throw new ForbiddenException("Access denied");
        }

        String folder = "tasks/" + taskId + "/attachments";
        String uploadUrl = s3StorageService.generateUploadUrl(folder, fileName);
        String s3Key = folder + "/" + UUID.randomUUID() + "-" + fileName;

        return AttachmentUrlResponse.builder()
            .uploadUrl(uploadUrl)
            .s3Key(s3Key)
            .expiresInSeconds(900)
            .build();
    }

    @Transactional
    public Response confirmAttachment(String taskId, String s3Key, String userEmail) {
        User user = getUserByEmail(userEmail);
        Task task = getTaskOrThrow(taskId);

        if (!task.getUserId().equals(user.getId())) {
            throw new ForbiddenException("Access denied");
        }

        task.getAttachmentKeys().add(s3Key);
        return toResponse(taskRepository.save(task));
    }

    private Response toResponse(Task task) {
        long unresolved = commentRepository.countByTaskIdAndResolved(task.getId(), false);

        List<String> attachmentUrls = task.getAttachmentKeys().stream()
            .map(s3StorageService::generateDownloadUrl)
            .collect(Collectors.toList());

        return Response.builder()
            .id(task.getId())
            .weekId(task.getWeekId())
            .userId(task.getUserId())
            .title(task.getTitle())
            .description(task.getDescription())
            .status(task.getStatus())
            .hoursSpent(task.getHoursSpent())
            .blocker(task.getBlocker())
            .evidenceLinks(task.getEvidenceLinks())
            .attachmentUrls(attachmentUrls)
            .priority(task.getPriority())
            .unresolvedComments(unresolved)
            .completedAt(task.getCompletedAt())
            .createdAt(task.getCreatedAt())
            .updatedAt(task.getUpdatedAt())
            .build();
    }

    private Task getTaskOrThrow(String id) {
        return taskRepository.findById(id)
            .orElseThrow(() -> new ResourceNotFoundException("Task", "id", id));
    }

    private Week getWeekOrThrow(String id) {
        return weekRepository.findById(id)
            .orElseThrow(() -> new ResourceNotFoundException("Week", "id", id));
    }

    private User getUserByEmail(String email) {
        return userRepository.findByEmail(email)
            .orElseThrow(() -> new ResourceNotFoundException("User", "email", email));
    }
}
