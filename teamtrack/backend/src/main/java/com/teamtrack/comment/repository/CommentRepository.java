package com.teamtrack.comment.repository;

import com.teamtrack.comment.model.Comment;
import com.teamtrack.comment.model.CommentType;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface CommentRepository extends MongoRepository<Comment, String> {
    List<Comment> findByTaskIdOrderByCreatedAtAsc(String taskId);
    List<Comment> findByTaskIdAndResolved(String taskId, boolean resolved);
    List<Comment> findByTaskIdAndType(String taskId, CommentType type);
    long countByTaskIdAndResolved(String taskId, boolean resolved);
    void deleteByTaskId(String taskId);
}
