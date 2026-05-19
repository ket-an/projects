package com.teamtrack.report.controller;

import com.teamtrack.report.dto.ReportDto.*;
import com.teamtrack.report.service.ReportService;
import com.teamtrack.util.ApiResponse;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.*;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/manager/reports")
@RequiredArgsConstructor
public class ReportController {

    private final ReportService reportService;

    @PostMapping("/generate")
    public ResponseEntity<ApiResponse<Response>> generateReport(
            @AuthenticationPrincipal UserDetails user,
            @Valid @RequestBody GenerateRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED)
            .body(ApiResponse.success("Report generation started",
                reportService.generateReport(user.getUsername(), request)));
    }

    @GetMapping
    public ResponseEntity<ApiResponse<List<Response>>> getReports(
            @AuthenticationPrincipal UserDetails user) {
        return ResponseEntity.ok(ApiResponse.success(
            reportService.getReports(user.getUsername())));
    }

    @GetMapping("/{id}/download")
    public ResponseEntity<ApiResponse<String>> getDownloadUrl(
            @PathVariable String id,
            @AuthenticationPrincipal UserDetails user) {
        String url = reportService.getDownloadUrl(id);
        return ResponseEntity.ok(ApiResponse.success("Download URL generated", url));
    }
}
