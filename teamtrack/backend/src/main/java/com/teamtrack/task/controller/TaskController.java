package com.teamtrack.task.controller;

import com.teamtrack.task.dto.TaskDto.*;
import com.teamtrack.task.service.TaskService;
import com.teamtrack.util.ApiResponse;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.*;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * Task Controller
 *
 * @DeleteMapping  - HTTP DELETE handler
 * @PatchMapping   - HTTP PATCH handler (partial update)
 * @RequestParam   - Query parameter binding
 */
@RestController
@RequestMapping("/tasks")
@RequiredArgsConstructor
public class TaskController {

    private final TaskService taskService;

    @PostMapping
    public ResponseEntity<ApiResponse<Response>> createTask(
            @AuthenticationPrincipal UserDetails user,
            @Valid @RequestBody CreateRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED)
            .body(ApiResponse.success("Task created",
                taskService.createTask(user.getUsername(), request)));
    }

    @GetMapping
    public ResponseEntity<ApiResponse<List<Response>>> getTasksByWeek(
            @RequestParam String weekId,
            @AuthenticationPrincipal UserDetails user) {
        return ResponseEntity.ok(ApiResponse.success(
            taskService.getTasksByWeek(weekId, user.getUsername())));
    }

    @PutMapping("/{id}")
    public ResponseEntity<ApiResponse<Response>> updateTask(
            @PathVariable String id,
            @AuthenticationPrincipal UserDetails user,
            @Valid @RequestBody UpdateRequest request) {
        return ResponseEntity.ok(ApiResponse.success("Task updated",
            taskService.updateTask(id, user.getUsername(), request)));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<ApiResponse<Void>> deleteTask(
            @PathVariable String id,
            @AuthenticationPrincipal UserDetails user) {
        taskService.deleteTask(id, user.getUsername());
        return ResponseEntity.ok(ApiResponse.success("Task deleted", null));
    }

    @PostMapping("/{id}/attachment-url")
    public ResponseEntity<ApiResponse<AttachmentUrlResponse>> getAttachmentUploadUrl(
            @PathVariable String id,
            @RequestParam String fileName,
            @AuthenticationPrincipal UserDetails user) {
        return ResponseEntity.ok(ApiResponse.success(
            taskService.generateAttachmentUploadUrl(id, fileName, user.getUsername())));
    }

    @PatchMapping("/{id}/attachment-confirm")
    public ResponseEntity<ApiResponse<Response>> confirmAttachment(
            @PathVariable String id,
            @RequestParam String s3Key,
            @AuthenticationPrincipal UserDetails user) {
        return ResponseEntity.ok(ApiResponse.success("Attachment confirmed",
            taskService.confirmAttachment(id, s3Key, user.getUsername())));
    }
}
