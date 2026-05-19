package com.teamtrack.report.model;

import lombok.*;
import org.springframework.data.annotation.*;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

import java.time.LocalDateTime;

@Document(collection = "reports")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Report {

    @Id
    private String id;

    @Field("manager_id")
    private String managerId;

    @Field("team_id")
    private String teamId;

    @Field("quarter")
    private String quarter; // Q1, Q2, Q3, Q4

    @Field("year")
    private int year;

    @Field("format")
    private ReportFormat format;

    @Field("s3_key")
    private String s3Key;

    @Field("file_name")
    private String fileName;

    @Field("generated_at")
    @CreatedDate
    private LocalDateTime generatedAt;
}
