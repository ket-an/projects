package com.teamtrack.task.model;

import lombok.*;
import org.springframework.data.annotation.*;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * Task MongoDB Document
 * Covers: @Document, @Id, @Field, @CreatedDate, @LastModifiedDate, @Builder.Default
 */
@Document(collection = "tasks")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Task {

    @Id
    private String id;

    @Field("week_id")
    private String weekId;

    @Field("user_id")
    private String userId;

    @Field("title")
    private String title;

    @Field("description")
    private String description;

    @Field("status")
    @Builder.Default
    private TaskStatus status = TaskStatus.TODO;

    @Field("hours_spent")
    @Builder.Default
    private double hoursSpent = 0;

    @Field("blocker")
    private String blocker;

    @Field("evidence_links")
    @Builder.Default
    private List<String> evidenceLinks = new ArrayList<>();

    @Field("attachment_keys")
    @Builder.Default
    private List<String> attachmentKeys = new ArrayList<>();

    @Field("completed_at")
    private LocalDateTime completedAt;

    @Field("priority")
    @Builder.Default
    private int priority = 2; // 1=High, 2=Medium, 3=Low

    @CreatedDate
    @Field("created_at")
    private LocalDateTime createdAt;

    @LastModifiedDate
    @Field("updated_at")
    private LocalDateTime updatedAt;
}
