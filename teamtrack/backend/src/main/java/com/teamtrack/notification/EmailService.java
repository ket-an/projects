package com.teamtrack.notification;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;

/**
 * Email Notification Service
 *
 * @Async - Runs method in a separate thread pool thread (non-blocking).
 *          Requires @EnableAsync on the main class.
 *          Email sending doesn't block the HTTP response thread.
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class EmailService {

    private final JavaMailSender mailSender;

    @Async
    public void sendWeekSubmittedNotification(String managerEmail, String memberName,
                                               String weekLabel) {
        String subject = String.format("[TeamTrack] %s submitted week: %s", memberName, weekLabel);
        String body = String.format(
            "Hello,\n\n%s has submitted their tasks for %s.\n\n" +
            "Please log in to TeamTrack to review and approve.\n\nTeamTrack System",
            memberName, weekLabel);
        sendSimpleEmail(managerEmail, subject, body);
    }

    @Async
    public void sendCommentNotification(String memberEmail, String managerName,
                                         String taskTitle, String commentType, String body) {
        String subject = String.format("[TeamTrack] New %s on your task: %s",
            commentType.toLowerCase(), taskTitle);
        String emailBody = String.format(
            "Hello,\n\n%s has added a %s on your task \"%s\":\n\n\"%s\"\n\n" +
            "Please log in to TeamTrack to view and respond.\n\nTeamTrack System",
            managerName, commentType.toLowerCase(), taskTitle, body);
        sendSimpleEmail(memberEmail, subject, emailBody);
    }

    @Async
    public void sendWeekApprovedNotification(String memberEmail, String weekLabel) {
        String subject = String.format("[TeamTrack] Your week %s has been approved!", weekLabel);
        String body = String.format(
            "Hello,\n\nYour weekly tasks for %s have been reviewed and approved.\n\n" +
            "Great work!\n\nTeamTrack System", weekLabel);
        sendSimpleEmail(memberEmail, subject, body);
    }

    @Async
    public void sendCommentResolvedNotification(String managerEmail, String memberName,
                                                  String taskTitle) {
        String subject = String.format("[TeamTrack] Comment resolved on: %s", taskTitle);
        String body = String.format(
            "Hello,\n\n%s has resolved your comment on task \"%s\".\n\n" +
            "TeamTrack System", memberName, taskTitle);
        sendSimpleEmail(managerEmail, subject, body);
    }

    @Async
    public void sendReportReadyNotification(String managerEmail, String quarter, int year) {
        String subject = String.format("[TeamTrack] Quarterly Report Ready: %s %d", quarter, year);
        String body = String.format(
            "Hello,\n\nYour quarterly work analysis report for %s %d has been generated.\n\n" +
            "Please log in to TeamTrack to download it.\n\nTeamTrack System", quarter, year);
        sendSimpleEmail(managerEmail, subject, body);
    }

    private void sendSimpleEmail(String to, String subject, String body) {
        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setTo(to);
            message.setSubject(subject);
            message.setText(body);
            mailSender.send(message);
            log.info("Email sent to {} — {}", to, subject);
        } catch (Exception e) {
            log.error("Failed to send email to {}: {}", to, e.getMessage());
        }
    }
}
