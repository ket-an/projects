package com.teamtrack.comment.model;

import lombok.*;
import org.springframework.data.annotation.*;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

import java.time.LocalDateTime;

@Document(collection = "comments")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Comment {

    @Id
    private String id;

    @Field("task_id")
    private String taskId;

    @Field("author_id")
    private String authorId;

    @Field("author_name")
    private String authorName;

    @Field("body")
    private String body;

    @Field("type")
    private CommentType type;

    @Field("resolved")
    @Builder.Default
    private boolean resolved = false;

    @Field("resolved_by_id")
    private String resolvedById;

    @Field("resolved_at")
    private LocalDateTime resolvedAt;

    @CreatedDate
    @Field("created_at")
    private LocalDateTime createdAt;

    @LastModifiedDate
    @Field("updated_at")
    private LocalDateTime updatedAt;
}
