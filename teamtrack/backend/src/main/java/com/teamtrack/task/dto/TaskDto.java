package com.teamtrack.task.dto;

import com.teamtrack.task.model.TaskStatus;
import jakarta.validation.constraints.*;
import lombok.*;
import java.time.LocalDateTime;
import java.util.List;

public class TaskDto {

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class CreateRequest {
        @NotBlank(message = "Title is required")
        @Size(max = 200)
        private String title;

        @NotBlank(message = "Description is required")
        private String description;

        @NotBlank(message = "Week ID is required")
        private String weekId;

        private TaskStatus status;

        @DecimalMin("0.0") @DecimalMax("24.0")
        private double hoursSpent;

        private String blocker;
        private List<String> evidenceLinks;

        @Min(1) @Max(3)
        private int priority = 2;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class UpdateRequest {
        @Size(max = 200)
        private String title;
        private String description;
        private TaskStatus status;
        @DecimalMin("0.0") @DecimalMax("24.0")
        private double hoursSpent;
        private String blocker;
        private List<String> evidenceLinks;
        @Min(1) @Max(3)
        private int priority;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class Response {
        private String id;
        private String weekId;
        private String userId;
        private String title;
        private String description;
        private TaskStatus status;
        private double hoursSpent;
        private String blocker;
        private List<String> evidenceLinks;
        private List<String> attachmentUrls;
        private int priority;
        private long unresolvedComments;
        private LocalDateTime completedAt;
        private LocalDateTime createdAt;
        private LocalDateTime updatedAt;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class AttachmentUrlResponse {
        private String uploadUrl;
        private String s3Key;
        private long expiresInSeconds;
    }
}
