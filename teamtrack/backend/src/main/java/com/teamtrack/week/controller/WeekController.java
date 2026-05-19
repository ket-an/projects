package com.teamtrack.week.controller;

import com.teamtrack.util.ApiResponse;
import com.teamtrack.week.dto.WeekDto.*;
import com.teamtrack.week.model.WeekStatus;
import com.teamtrack.week.service.WeekService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.http.*;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * Week Controller
 *
 * @PathVariable   - Binds a URI template variable {id} to a method parameter
 * @RequestParam   - Binds a query string parameter (e.g. ?status=SUBMITTED)
 * @ResponseStatus - Sets the default HTTP response status for the method
 */
@RestController
@RequiredArgsConstructor
public class WeekController {

    private final WeekService weekService;

    @PostMapping("/weeks")
    @ResponseStatus(HttpStatus.CREATED)
    public ResponseEntity<ApiResponse<Response>> createWeek(
            @AuthenticationPrincipal UserDetails user,
            @Valid @RequestBody CreateRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED)
            .body(ApiResponse.success("Week created", weekService.createWeek(user.getUsername(), request)));
    }

    @GetMapping("/weeks/my")
    public ResponseEntity<ApiResponse<List<Response>>> getMyWeeks(
            @AuthenticationPrincipal UserDetails user) {
        return ResponseEntity.ok(ApiResponse.success(weekService.getMyWeeks(user.getUsername())));
    }

    @GetMapping("/weeks/{id}")
    public ResponseEntity<ApiResponse<Response>> getWeek(
            @PathVariable String id,
            @AuthenticationPrincipal UserDetails user) {
        return ResponseEntity.ok(ApiResponse.success(
            weekService.getWeekById(id, user.getUsername())));
    }

    @PutMapping("/weeks/{id}/submit")
    public ResponseEntity<ApiResponse<Response>> submitWeek(
            @PathVariable String id,
            @AuthenticationPrincipal UserDetails user,
            @RequestBody(required = false) SubmitRequest request) {
        return ResponseEntity.ok(ApiResponse.success("Week submitted",
            weekService.submitWeek(id, user.getUsername(), request)));
    }

    // Manager endpoints
    @GetMapping("/manager/weeks")
    public ResponseEntity<ApiResponse<Page<Response>>> getAllWeeks(
            @RequestParam(required = false) WeekStatus status,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size,
            @AuthenticationPrincipal UserDetails user) {
        PageRequest pageable = PageRequest.of(page, size, Sort.by("updatedAt").descending());
        return ResponseEntity.ok(ApiResponse.success(
            weekService.getAllWeeksForManager(status, pageable)));
    }

    @PutMapping("/manager/weeks/{id}/approve")
    public ResponseEntity<ApiResponse<Response>> approveWeek(
            @PathVariable String id,
            @AuthenticationPrincipal UserDetails user) {
        return ResponseEntity.ok(ApiResponse.success("Week approved",
            weekService.approveWeek(id, user.getUsername())));
    }
}
