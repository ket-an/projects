// ─── WeekRepository ─────────────────────────────────────────────────────────
// File: week/repository/WeekRepository.java
package com.teamtrack.week.repository;

import com.teamtrack.week.model.Week;
import com.teamtrack.week.model.WeekStatus;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

@Repository
public interface WeekRepository extends MongoRepository<Week, String> {
    List<Week> findByUserId(String userId);
    List<Week> findByUserIdOrderByStartDateDesc(String userId);
    List<Week> findByUserIdAndStatus(String userId, WeekStatus status);
    Page<Week> findByStatus(WeekStatus status, Pageable pageable);
    Optional<Week> findByUserIdAndStartDateAndEndDate(String userId, LocalDate start, LocalDate end);
    boolean existsByUserIdAndWeekLabel(String userId, String weekLabel);
    long countByUserIdAndStatus(String userId, WeekStatus status);

    @Query("{ 'start_date': { $gte: ?0 }, 'end_date': { $lte: ?1 } }")
    List<Week> findByDateRange(LocalDate from, LocalDate to);
}
