package com.teamtrack.week.service;

import com.teamtrack.auth.model.User;
import com.teamtrack.auth.repository.UserRepository;
import com.teamtrack.auth.service.AuthService;
import com.teamtrack.exception.BadRequestException;
import com.teamtrack.exception.ForbiddenException;
import com.teamtrack.exception.ResourceNotFoundException;
import com.teamtrack.notification.EmailService;
import com.teamtrack.task.model.Task;
import com.teamtrack.task.model.TaskStatus;
import com.teamtrack.task.repository.TaskRepository;
import com.teamtrack.util.Role;
import com.teamtrack.week.dto.WeekDto.*;
import com.teamtrack.week.model.Week;
import com.teamtrack.week.model.WeekStatus;
import com.teamtrack.week.repository.WeekRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.data.domain.Pageable;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Week Service
 *
 * @PreAuthorize  - Method-level security check evaluated before the method executes.
 *                  Uses Spring EL: "#userId" refers to method parameter.
 * @Cacheable     - Caches the return value; subsequent calls return cached data without executing.
 * @CacheEvict    - Removes entry from cache when data changes.
 * @Transactional - Wraps in transaction; rolls back on any RuntimeException.
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class WeekService {

    private final WeekRepository weekRepository;
    private final UserRepository userRepository;
    private final TaskRepository taskRepository;
    private final EmailService emailService;

    @Transactional
    public Response createWeek(String userEmail, CreateRequest request) {
        User user = getUserByEmail(userEmail);

        if (weekRepository.existsByUserIdAndWeekLabel(user.getId(), request.getWeekLabel())) {
            throw new BadRequestException("Week '" + request.getWeekLabel() + "' already exists");
        }

        Week week = Week.builder()
            .userId(user.getId())
            .weekLabel(request.getWeekLabel())
            .startDate(request.getStartDate())
            .endDate(request.getEndDate())
            .status(WeekStatus.DRAFT)
            .build();

        return toResponse(weekRepository.save(week), user);
    }

    @Transactional(readOnly = true)
    public List<Response> getMyWeeks(String userEmail) {
        User user = getUserByEmail(userEmail);
        return weekRepository.findByUserIdOrderByStartDateDesc(user.getId())
            .stream().map(w -> toResponse(w, user)).collect(Collectors.toList());
    }

    @Transactional(readOnly = true)
    public Response getWeekById(String weekId, String userEmail) {
        User user = getUserByEmail(userEmail);
        Week week = getWeekOrThrow(weekId);

        // Members can only see their own weeks; managers can see all
        if (user.getRole() == Role.TEAM_MEMBER && !week.getUserId().equals(user.getId())) {
            throw new ForbiddenException("You can only view your own weeks");
        }

        User weekOwner = userRepository.findById(week.getUserId())
            .orElseThrow(() -> new ResourceNotFoundException("User", "id", week.getUserId()));
        return toResponse(week, weekOwner);
    }

    @Transactional
    @CacheEvict(value = "weekStats", key = "#userEmail")
    public Response submitWeek(String weekId, String userEmail, SubmitRequest request) {
        User user = getUserByEmail(userEmail);
        Week week = getWeekOrThrow(weekId);

        if (!week.getUserId().equals(user.getId())) {
            throw new ForbiddenException("You can only submit your own weeks");
        }
        if (week.getStatus() != WeekStatus.DRAFT) {
            throw new BadRequestException("Only DRAFT weeks can be submitted");
        }

        long taskCount = taskRepository.countByWeekId(weekId);
        if (taskCount == 0) {
            throw new BadRequestException("Cannot submit a week with no tasks");
        }

        // Sync stats before submitting
        List<Task> tasks = taskRepository.findByWeekId(weekId);
        week.setTotalTasks((int) taskCount);
        week.setCompletedTasks((int) tasks.stream()
            .filter(t -> t.getStatus() == TaskStatus.COMPLETED).count());
        week.setTotalHours(tasks.stream().mapToDouble(Task::getHoursSpent).sum());
        week.setStatus(WeekStatus.SUBMITTED);
        if (request != null) week.setSubmissionNote(request.getSubmissionNote());

        Week saved = weekRepository.save(week);
        log.info("Week {} submitted by {}", weekId, userEmail);

        // Notify managers asynchronously via @Async in EmailService
        userRepository.findByRole(Role.MANAGER).forEach(manager ->
            emailService.sendWeekSubmittedNotification(
                manager.getEmail(), user.getName(), week.getWeekLabel()));

        return toResponse(saved, user);
    }

    /**
     * @PreAuthorize - Restricts this method to MANAGER role.
     *                 Evaluated by Spring AOP before method invocation.
     */
    @PreAuthorize("hasRole('MANAGER')")
    @Transactional
    public Response approveWeek(String weekId, String managerEmail) {
        User manager = getUserByEmail(managerEmail);
        Week week = getWeekOrThrow(weekId);

        if (week.getStatus() != WeekStatus.SUBMITTED) {
            throw new BadRequestException("Only SUBMITTED weeks can be approved");
        }

        week.setStatus(WeekStatus.APPROVED);
        week.setApprovedBy(manager.getName());
        week.setApprovedAt(LocalDateTime.now());

        Week saved = weekRepository.save(week);
        log.info("Week {} approved by {}", weekId, managerEmail);

        User member = userRepository.findById(week.getUserId()).orElseThrow();
        emailService.sendWeekApprovedNotification(member.getEmail(), week.getWeekLabel());

        return toResponse(saved, member);
    }

    @PreAuthorize("hasRole('MANAGER')")
    @Transactional(readOnly = true)
    public Page<Response> getAllWeeksForManager(WeekStatus status, Pageable pageable) {
        Page<Week> weeks = (status != null)
            ? weekRepository.findByStatus(status, pageable)
            : weekRepository.findAll(pageable);

        List<Response> responses = weeks.getContent().stream().map(week -> {
            User owner = userRepository.findById(week.getUserId())
                .orElse(User.builder().name("Unknown").build());
            return toResponse(week, owner);
        }).collect(Collectors.toList());

        return new PageImpl<>(responses, pageable, weeks.getTotalElements());
    }

    private Week getWeekOrThrow(String id) {
        return weekRepository.findById(id)
            .orElseThrow(() -> new ResourceNotFoundException("Week", "id", id));
    }

    private User getUserByEmail(String email) {
        return userRepository.findByEmail(email)
            .orElseThrow(() -> new ResourceNotFoundException("User", "email", email));
    }

    private Response toResponse(Week week, User owner) {
        return Response.builder()
            .id(week.getId())
            .userId(week.getUserId())
            .userName(owner.getName())
            .weekLabel(week.getWeekLabel())
            .startDate(week.getStartDate())
            .endDate(week.getEndDate())
            .status(week.getStatus())
            .totalTasks(week.getTotalTasks())
            .completedTasks(week.getCompletedTasks())
            .totalHours(week.getTotalHours())
            .submissionNote(week.getSubmissionNote())
            .approvedBy(week.getApprovedBy())
            .approvedAt(week.getApprovedAt())
            .createdAt(week.getCreatedAt())
            .updatedAt(week.getUpdatedAt())
            .build();
    }
}
