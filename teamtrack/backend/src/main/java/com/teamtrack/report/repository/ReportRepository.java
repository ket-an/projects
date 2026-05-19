package com.teamtrack.report.repository;

import com.teamtrack.report.model.Report;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ReportRepository extends MongoRepository<Report, String> {
    List<Report> findByManagerIdOrderByGeneratedAtDesc(String managerId);
    List<Report> findByTeamIdAndYearAndQuarter(String teamId, int year, String quarter);
}
