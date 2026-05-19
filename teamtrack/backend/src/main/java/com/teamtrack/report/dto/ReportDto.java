package com.teamtrack.report.dto;

import com.teamtrack.report.model.ReportFormat;
import jakarta.validation.constraints.*;
import lombok.*;
import java.time.LocalDateTime;

public class ReportDto {

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class GenerateRequest {
        @NotBlank private String teamId;
        @NotBlank @Pattern(regexp = "Q[1-4]") private String quarter;
        @Min(2020) @Max(2099) private int year;
        @NotNull private ReportFormat format;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class Response {
        private String id;
        private String teamId;
        private String quarter;
        private int year;
        private ReportFormat format;
        private String fileName;
        private String downloadUrl;
        private LocalDateTime generatedAt;
    }
}
