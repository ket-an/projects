package com.teamtrack.week.dto;

import com.teamtrack.week.model.WeekStatus;
import jakarta.validation.constraints.*;
import lombok.*;
import java.time.LocalDate;
import java.time.LocalDateTime;

public class WeekDto {

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class CreateRequest {
        @NotBlank private String weekLabel;
        @NotNull private LocalDate startDate;
        @NotNull private LocalDate endDate;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class SubmitRequest {
        private String submissionNote;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class Response {
        private String id;
        private String userId;
        private String userName;
        private String weekLabel;
        private LocalDate startDate;
        private LocalDate endDate;
        private WeekStatus status;
        private int totalTasks;
        private int completedTasks;
        private double totalHours;
        private String submissionNote;
        private String approvedBy;
        private LocalDateTime approvedAt;
        private LocalDateTime createdAt;
        private LocalDateTime updatedAt;
    }
}
