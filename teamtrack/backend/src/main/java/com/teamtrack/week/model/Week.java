package com.teamtrack.week.model;

import lombok.*;
import org.springframework.data.annotation.*;
import org.springframework.data.mongodb.core.index.CompoundIndex;
import org.springframework.data.mongodb.core.index.CompoundIndexes;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

import java.time.LocalDate;
import java.time.LocalDateTime;

/**
 * Week MongoDB Document
 *
 * @CompoundIndex  - Defines a multi-field index on the collection for query performance
 * @CompoundIndexes - Container for multiple @CompoundIndex declarations
 */
@Document(collection = "weeks")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@CompoundIndexes({
    @CompoundIndex(name = "user_status_idx", def = "{'user_id': 1, 'status': 1}"),
    @CompoundIndex(name = "date_range_idx", def = "{'start_date': 1, 'end_date': 1}")
})
public class Week {

    @Id
    private String id;

    @Field("user_id")
    private String userId;

    @Field("week_label")
    private String weekLabel;

    @Field("start_date")
    private LocalDate startDate;

    @Field("end_date")
    private LocalDate endDate;

    @Field("status")
    @Builder.Default
    private WeekStatus status = WeekStatus.DRAFT;

    @Field("total_tasks")
    @Builder.Default
    private int totalTasks = 0;

    @Field("completed_tasks")
    @Builder.Default
    private int completedTasks = 0;

    @Field("total_hours")
    @Builder.Default
    private double totalHours = 0;

    @Field("approved_by")
    private String approvedBy;

    @Field("approved_at")
    private LocalDateTime approvedAt;

    @Field("submission_note")
    private String submissionNote;

    @CreatedDate
    @Field("created_at")
    private LocalDateTime createdAt;

    @LastModifiedDate
    @Field("updated_at")
    private LocalDateTime updatedAt;
}
