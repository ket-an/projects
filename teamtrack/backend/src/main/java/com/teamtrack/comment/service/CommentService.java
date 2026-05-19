package com.teamtrack.comment.service;

import com.teamtrack.auth.model.User;
import com.teamtrack.auth.repository.UserRepository;
import com.teamtrack.comment.dto.CommentDto.*;
import com.teamtrack.comment.model.Comment;
import com.teamtrack.comment.repository.CommentRepository;
import com.teamtrack.exception.*;
import com.teamtrack.notification.EmailService;
import com.teamtrack.task.model.Task;
import com.teamtrack.task.repository.TaskRepository;
import com.teamtrack.util.Role;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class CommentService {

    private final CommentRepository commentRepository;
    private final TaskRepository taskRepository;
    private final UserRepository userRepository;
    private final EmailService emailService;

    @PreAuthorize("hasRole('MANAGER')")
    @Transactional
    public Response addComment(String managerEmail, CreateRequest request) {
        User manager = getUserByEmail(managerEmail);
        Task task = taskRepository.findById(request.getTaskId())
            .orElseThrow(() -> new ResourceNotFoundException("Task", "id", request.getTaskId()));

        Comment comment = Comment.builder()
            .taskId(request.getTaskId())
            .authorId(manager.getId())
            .authorName(manager.getName())
            .body(request.getBody())
            .type(request.getType())
            .resolved(false)
            .build();

        Comment saved = commentRepository.save(comment);
        log.info("Comment {} added by manager {} on task {}", saved.getId(), managerEmail, request.getTaskId());

        // Notify task owner asynchronously
        userRepository.findById(task.getUserId()).ifPresent(member ->
            emailService.sendCommentNotification(
                member.getEmail(), manager.getName(), task.getTitle(),
                request.getType().name(), request.getBody()));

        return toResponse(saved);
    }

    @Transactional(readOnly = true)
    public List<Response> getCommentsByTask(String taskId) {
        return commentRepository.findByTaskIdOrderByCreatedAtAsc(taskId)
            .stream().map(this::toResponse).collect(Collectors.toList());
    }

    @Transactional
    public Response resolveComment(String commentId, String memberEmail) {
        User member = getUserByEmail(memberEmail);

        if (member.getRole() != Role.TEAM_MEMBER) {
            throw new ForbiddenException("Only team members can resolve comments");
        }

        Comment comment = commentRepository.findById(commentId)
            .orElseThrow(() -> new ResourceNotFoundException("Comment", "id", commentId));

        if (comment.isResolved()) {
            throw new BadRequestException("Comment is already resolved");
        }

        // Verify the task belongs to this member
        Task task = taskRepository.findById(comment.getTaskId())
            .orElseThrow(() -> new ResourceNotFoundException("Task", "id", comment.getTaskId()));

        if (!task.getUserId().equals(member.getId())) {
            throw new ForbiddenException("You can only resolve comments on your own tasks");
        }

        comment.setResolved(true);
        comment.setResolvedById(member.getId());
        comment.setResolvedAt(LocalDateTime.now());

        Comment saved = commentRepository.save(comment);
        log.info("Comment {} resolved by {}", commentId, memberEmail);

        // Notify the manager
        userRepository.findById(comment.getAuthorId()).ifPresent(manager ->
            emailService.sendCommentResolvedNotification(
                manager.getEmail(), member.getName(), task.getTitle()));

        return toResponse(saved);
    }

    private Response toResponse(Comment c) {
        return Response.builder()
            .id(c.getId())
            .taskId(c.getTaskId())
            .authorId(c.getAuthorId())
            .authorName(c.getAuthorName())
            .body(c.getBody())
            .type(c.getType())
            .resolved(c.isResolved())
            .resolvedById(c.getResolvedById())
            .resolvedAt(c.getResolvedAt())
            .createdAt(c.getCreatedAt())
            .build();
    }

    private User getUserByEmail(String email) {
        return userRepository.findByEmail(email)
            .orElseThrow(() -> new ResourceNotFoundException("User", "email", email));
    }
}
