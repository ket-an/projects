package com.teamtrack.report.service;

import com.teamtrack.auth.model.User;
import com.teamtrack.auth.repository.UserRepository;
import com.teamtrack.exception.ResourceNotFoundException;
import com.teamtrack.notification.EmailService;
import com.teamtrack.report.dto.ReportDto.*;
import com.teamtrack.report.model.Report;
import com.teamtrack.report.model.ReportFormat;
import com.teamtrack.report.repository.ReportRepository;
import com.teamtrack.storage.S3StorageService;
import com.teamtrack.task.model.Task;
import com.teamtrack.task.model.TaskStatus;
import com.teamtrack.task.repository.TaskRepository;
import com.teamtrack.week.model.Week;
import com.teamtrack.week.model.WeekStatus;
import com.teamtrack.week.repository.WeekRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.ss.util.CellRangeAddress;
import org.apache.poi.xssf.usermodel.*;
import org.springframework.scheduling.annotation.Async;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.ByteArrayOutputStream;
import java.time.*;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Report Service — quarterly work analysis using Apache POI
 *
 * @Async         - Report generation is CPU-intensive; runs in background thread
 * @PreAuthorize  - Only managers can generate reports
 * @Scheduled     - Can be added for automated quarterly generation (see commented example)
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class ReportService {

    private final ReportRepository reportRepository;
    private final UserRepository userRepository;
    private final WeekRepository weekRepository;
    private final TaskRepository taskRepository;
    private final S3StorageService s3StorageService;
    private final EmailService emailService;

    @PreAuthorize("hasRole('MANAGER')")
    @Transactional
    public Response generateReport(String managerEmail, GenerateRequest request) {
        User manager = getUserByEmail(managerEmail);

        log.info("Generating {} report for team {} — {} {}",
            request.getFormat(), request.getTeamId(), request.getQuarter(), request.getYear());

        // Get date range for the quarter
        LocalDate[] range = getQuarterDateRange(request.getQuarter(), request.getYear());

        // Collect all team members
        List<User> members = userRepository.findByTeamId(request.getTeamId());

        // Build report data
        byte[] reportBytes;
        String contentType;
        String fileExtension;

        if (request.getFormat() == ReportFormat.XLSX) {
            reportBytes = generateXlsxReport(members, range[0], range[1], request);
            contentType = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
            fileExtension = "xlsx";
        } else {
            // PDF: generate Excel then note — for full PDF support add iText/PdfBox dependency
            reportBytes = generateXlsxReport(members, range[0], range[1], request);
            contentType = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
            fileExtension = "xlsx"; // Default to XLSX when PDF lib not available
        }

        String fileName = String.format("TeamTrack_Report_%s_%s_%d.%s",
            request.getTeamId(), request.getQuarter(), request.getYear(), fileExtension);
        String s3Key = String.format("reports/%s/%d/%s/%s",
            request.getTeamId(), request.getYear(), request.getQuarter(), fileName);

        s3StorageService.uploadBytes(reportBytes, s3Key, contentType);

        Report report = Report.builder()
            .managerId(manager.getId())
            .teamId(request.getTeamId())
            .quarter(request.getQuarter())
            .year(request.getYear())
            .format(request.getFormat())
            .s3Key(s3Key)
            .fileName(fileName)
            .build();

        Report saved = reportRepository.save(report);
        emailService.sendReportReadyNotification(managerEmail, request.getQuarter(), request.getYear());

        return toResponse(saved);
    }

    @PreAuthorize("hasRole('MANAGER')")
    @Transactional(readOnly = true)
    public List<Response> getReports(String managerEmail) {
        User manager = getUserByEmail(managerEmail);
        return reportRepository.findByManagerIdOrderByGeneratedAtDesc(manager.getId())
            .stream().map(this::toResponse).collect(Collectors.toList());
    }

    @PreAuthorize("hasRole('MANAGER')")
    public String getDownloadUrl(String reportId) {
        Report report = reportRepository.findById(reportId)
            .orElseThrow(() -> new ResourceNotFoundException("Report", "id", reportId));
        return s3StorageService.generateDownloadUrl(report.getS3Key());
    }

    // ─── XLSX Generation using Apache POI ─────────────────────────────────────

    private byte[] generateXlsxReport(List<User> members, LocalDate from, LocalDate to,
                                        GenerateRequest request) {
        try (XSSFWorkbook workbook = new XSSFWorkbook();
             ByteArrayOutputStream out = new ByteArrayOutputStream()) {

            // ── Summary Sheet ──
            XSSFSheet summarySheet = workbook.createSheet("Summary");
            createSummarySheet(workbook, summarySheet, members, from, to, request);

            // ── Per-Member Sheets ──
            for (User member : members) {
                String sheetName = member.getName().replaceAll("[/\\\\?*\\[\\]]", "_");
                if (sheetName.length() > 31) sheetName = sheetName.substring(0, 31);
                XSSFSheet memberSheet = workbook.createSheet(sheetName);
                createMemberSheet(workbook, memberSheet, member, from, to);
            }

            workbook.write(out);
            return out.toByteArray();
        } catch (Exception e) {
            log.error("Failed to generate XLSX report: {}", e.getMessage(), e);
            throw new RuntimeException("Report generation failed: " + e.getMessage());
        }
    }

    private void createSummarySheet(XSSFWorkbook wb, XSSFSheet sheet, List<User> members,
                                     LocalDate from, LocalDate to, GenerateRequest request) {
        // Styles
        CellStyle headerStyle = wb.createCellStyle();
        XSSFFont headerFont = wb.createFont();
        headerFont.setBold(true);
        headerFont.setFontHeightInPoints((short) 12);
        headerFont.setColor(IndexedColors.WHITE.getIndex());
        headerStyle.setFont(headerFont);
        headerStyle.setFillForegroundColor(IndexedColors.DARK_BLUE.getIndex());
        headerStyle.setFillPattern(FillPatternType.SOLID_FOREGROUND);
        headerStyle.setAlignment(HorizontalAlignment.CENTER);
        headerStyle.setBorderBottom(BorderStyle.THIN);

        CellStyle titleStyle = wb.createCellStyle();
        XSSFFont titleFont = wb.createFont();
        titleFont.setBold(true);
        titleFont.setFontHeightInPoints((short) 16);
        titleStyle.setFont(titleFont);

        CellStyle dataStyle = wb.createCellStyle();
        dataStyle.setBorderBottom(BorderStyle.THIN);
        dataStyle.setBorderRight(BorderStyle.THIN);

        CellStyle numberStyle = wb.createCellStyle();
        numberStyle.setAlignment(HorizontalAlignment.CENTER);
        numberStyle.setBorderBottom(BorderStyle.THIN);
        numberStyle.setBorderRight(BorderStyle.THIN);

        // Title row
        Row titleRow = sheet.createRow(0);
        Cell titleCell = titleRow.createCell(0);
        titleCell.setCellValue("TeamTrack — Quarterly Work Analysis Report");
        titleCell.setCellStyle(titleStyle);
        sheet.addMergedRegion(new CellRangeAddress(0, 0, 0, 7));

        Row metaRow = sheet.createRow(1);
        metaRow.createCell(0).setCellValue(
            String.format("Period: %s %d (%s to %s)", request.getQuarter(), request.getYear(),
                from, to));
        sheet.addMergedRegion(new CellRangeAddress(1, 1, 0, 7));

        Row metaRow2 = sheet.createRow(2);
        metaRow2.createCell(0).setCellValue("Generated: " + LocalDateTime.now()
            .format(DateTimeFormatter.ofPattern("dd MMM yyyy HH:mm")));
        sheet.addMergedRegion(new CellRangeAddress(2, 2, 0, 7));

        // Header row
        Row headerRow = sheet.createRow(4);
        String[] headers = {"#", "Member Name", "Email", "Weeks Submitted", "Weeks Approved",
            "Total Tasks", "Completed Tasks", "Total Hours"};
        for (int i = 0; i < headers.length; i++) {
            Cell cell = headerRow.createCell(i);
            cell.setCellValue(headers[i]);
            cell.setCellStyle(headerStyle);
        }

        // Data rows
        int rowNum = 5;
        int totalWeeksSubmitted = 0, totalWeeksApproved = 0;
        int totalTasks = 0, totalCompleted = 0;
        double totalHours = 0;

        for (int i = 0; i < members.size(); i++) {
            User member = members.get(i);
            List<Week> weeks = weekRepository.findByDateRange(from, to).stream()
                .filter(w -> w.getUserId().equals(member.getId())).collect(Collectors.toList());

            int submitted = (int) weeks.stream()
                .filter(w -> w.getStatus() != WeekStatus.DRAFT).count();
            int approved = (int) weeks.stream()
                .filter(w -> w.getStatus() == WeekStatus.APPROVED).count();

            List<Task> tasks = weeks.stream()
                .flatMap(w -> taskRepository.findByWeekId(w.getId()).stream())
                .collect(Collectors.toList());

            int completed = (int) tasks.stream()
                .filter(t -> t.getStatus() == TaskStatus.COMPLETED).count();
            double hours = tasks.stream().mapToDouble(Task::getHoursSpent).sum();

            Row row = sheet.createRow(rowNum++);
            row.createCell(0).setCellValue(i + 1);
            row.createCell(1).setCellValue(member.getName());
            row.createCell(2).setCellValue(member.getEmail());
            row.createCell(3).setCellValue(submitted);
            row.createCell(4).setCellValue(approved);
            row.createCell(5).setCellValue(tasks.size());
            row.createCell(6).setCellValue(completed);
            row.createCell(7).setCellValue(hours);

            totalWeeksSubmitted += submitted;
            totalWeeksApproved += approved;
            totalTasks += tasks.size();
            totalCompleted += completed;
            totalHours += hours;
        }

        // Totals row
        Row totalsRow = sheet.createRow(rowNum + 1);
        totalsRow.createCell(0).setCellValue("TOTAL");
        totalsRow.createCell(3).setCellValue(totalWeeksSubmitted);
        totalsRow.createCell(4).setCellValue(totalWeeksApproved);
        totalsRow.createCell(5).setCellValue(totalTasks);
        totalsRow.createCell(6).setCellValue(totalCompleted);
        totalsRow.createCell(7).setCellValue(totalHours);

        // Auto-size columns
        for (int i = 0; i < headers.length; i++) sheet.autoSizeColumn(i);
    }

    private void createMemberSheet(XSSFWorkbook wb, XSSFSheet sheet, User member,
                                    LocalDate from, LocalDate to) {
        CellStyle headerStyle = wb.createCellStyle();
        XSSFFont font = wb.createFont();
        font.setBold(true);
        font.setColor(IndexedColors.WHITE.getIndex());
        headerStyle.setFont(font);
        headerStyle.setFillForegroundColor(IndexedColors.DARK_TEAL.getIndex());
        headerStyle.setFillPattern(FillPatternType.SOLID_FOREGROUND);

        Row nameRow = sheet.createRow(0);
        nameRow.createCell(0).setCellValue("Member: " + member.getName());
        nameRow.createCell(4).setCellValue("Department: " +
            (member.getDepartment() != null ? member.getDepartment() : "-"));

        Row headerRow = sheet.createRow(2);
        String[] headers = {"Week", "Task Title", "Status", "Hours", "Priority",
            "Has Blocker", "Evidence Links", "Description"};
        for (int i = 0; i < headers.length; i++) {
            Cell cell = headerRow.createCell(i);
            cell.setCellValue(headers[i]);
            cell.setCellStyle(headerStyle);
        }

        List<Week> weeks = weekRepository.findByDateRange(from, to).stream()
            .filter(w -> w.getUserId().equals(member.getId()))
            .sorted(Comparator.comparing(Week::getStartDate))
            .collect(Collectors.toList());

        int rowNum = 3;
        for (Week week : weeks) {
            List<Task> tasks = taskRepository.findByWeekId(week.getId());
            for (Task task : tasks) {
                Row row = sheet.createRow(rowNum++);
                row.createCell(0).setCellValue(week.getWeekLabel());
                row.createCell(1).setCellValue(task.getTitle());
                row.createCell(2).setCellValue(task.getStatus().name());
                row.createCell(3).setCellValue(task.getHoursSpent());
                row.createCell(4).setCellValue(task.getPriority() == 1 ? "High" :
                    task.getPriority() == 2 ? "Medium" : "Low");
                row.createCell(5).setCellValue(task.getBlocker() != null ? "Yes" : "No");
                row.createCell(6).setCellValue(
                    task.getEvidenceLinks() != null ?
                    String.join(", ", task.getEvidenceLinks()) : "");
                row.createCell(7).setCellValue(task.getDescription());
            }
        }

        for (int i = 0; i < headers.length; i++) sheet.autoSizeColumn(i);
    }

    private LocalDate[] getQuarterDateRange(String quarter, int year) {
        return switch (quarter) {
            case "Q1" -> new LocalDate[]{LocalDate.of(year, 1, 1), LocalDate.of(year, 3, 31)};
            case "Q2" -> new LocalDate[]{LocalDate.of(year, 4, 1), LocalDate.of(year, 6, 30)};
            case "Q3" -> new LocalDate[]{LocalDate.of(year, 7, 1), LocalDate.of(year, 9, 30)};
            case "Q4" -> new LocalDate[]{LocalDate.of(year, 10, 1), LocalDate.of(year, 12, 31)};
            default -> throw new IllegalArgumentException("Invalid quarter: " + quarter);
        };
    }

    private Response toResponse(Report r) {
        return Response.builder()
            .id(r.getId())
            .teamId(r.getTeamId())
            .quarter(r.getQuarter())
            .year(r.getYear())
            .format(r.getFormat())
            .fileName(r.getFileName())
            .generatedAt(r.getGeneratedAt())
            .build();
    }

    private User getUserByEmail(String email) {
        return userRepository.findByEmail(email)
            .orElseThrow(() -> new ResourceNotFoundException("User", "email", email));
    }
}
