package com.teamtrack.task.repository;

import com.teamtrack.task.model.Task;
import com.teamtrack.task.model.TaskStatus;
import org.springframework.data.mongodb.repository.Aggregation;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * Task Repository
 *
 * @Aggregation - Runs a MongoDB aggregation pipeline directly from a repository method.
 *               Powerful for grouped summaries like total hours per user.
 */
@Repository
public interface TaskRepository extends MongoRepository<Task, String> {
    List<Task> findByWeekId(String weekId);
    List<Task> findByWeekIdAndUserId(String weekId, String userId);
    List<Task> findByUserId(String userId);
    List<Task> findByUserIdAndStatus(String userId, TaskStatus status);
    long countByWeekId(String weekId);
    long countByWeekIdAndStatus(String weekId, TaskStatus status);
    void deleteByWeekId(String weekId);

    @Aggregation(pipeline = {
        "{ $match: { 'user_id': ?0 } }",
        "{ $group: { _id: null, totalHours: { $sum: '$hours_spent' } } }"
    })
    Double sumHoursByUserId(String userId);
}
