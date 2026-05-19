package com.teamtrack.comment.dto;

import com.teamtrack.comment.model.CommentType;
import jakarta.validation.constraints.*;
import lombok.*;
import java.time.LocalDateTime;

public class CommentDto {

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class CreateRequest {
        @NotBlank(message = "Task ID is required")
        private String taskId;

        @NotBlank(message = "Comment body is required")
        @Size(max = 2000)
        private String body;

        @NotNull(message = "Comment type is required")
        private CommentType type;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class Response {
        private String id;
        private String taskId;
        private String authorId;
        private String authorName;
        private String body;
        private CommentType type;
        private boolean resolved;
        private String resolvedById;
        private LocalDateTime resolvedAt;
        private LocalDateTime createdAt;
    }
}
