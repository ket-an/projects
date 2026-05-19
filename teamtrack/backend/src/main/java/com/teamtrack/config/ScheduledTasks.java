package com.teamtrack.config;

import com.teamtrack.report.repository.ReportRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;

/**
 * Scheduled Tasks
 *
 * @Scheduled - Marks a method for periodic execution. Requires @EnableScheduling.
 *
 * Scheduling options:
 *   fixedRate        = run every N milliseconds (from start of last execution)
 *   fixedDelay       = run every N milliseconds (from end of last execution)
 *   cron             = cron expression "s m h d M DOW"
 *   initialDelay     = wait N ms before first execution
 *
 * Cron format: "0 0 2 * * *" = every day at 2:00 AM
 * Cron format: "0 0 0 1 1/3 *" = first day of every quarter
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ScheduledTasks {

    private final ReportRepository reportRepository;

    /**
     * Runs at 2:00 AM every Sunday — logs report stats (example scheduled task)
     */
    @Scheduled(cron = "0 0 2 * * SUN")
    public void weeklyReportStats() {
        long count = reportRepository.count();
        log.info("[Scheduled] Weekly stats — total reports generated: {}", count);
    }

    /**
     * Runs every 6 hours — health check ping
     */
    @Scheduled(fixedRate = 21_600_000, initialDelay = 60_000)
    public void healthPing() {
        log.debug("[Scheduled] Application health ping — {}", LocalDateTime.now());
    }
}
