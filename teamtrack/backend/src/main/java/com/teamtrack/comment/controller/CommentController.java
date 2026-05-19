package com.teamtrack.comment.controller;

import com.teamtrack.comment.dto.CommentDto.*;
import com.teamtrack.comment.service.CommentService;
import com.teamtrack.util.ApiResponse;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.*;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/comments")
@RequiredArgsConstructor
public class CommentController {

    private final CommentService commentService;

    /** Manager adds a comment */
    @PostMapping
    public ResponseEntity<ApiResponse<Response>> addComment(
            @AuthenticationPrincipal UserDetails user,
            @Valid @RequestBody CreateRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED)
            .body(ApiResponse.success("Comment added",
                commentService.addComment(user.getUsername(), request)));
    }

    /** Anyone can view comments on a task */
    @GetMapping
    public ResponseEntity<ApiResponse<List<Response>>> getComments(
            @RequestParam String taskId) {
        return ResponseEntity.ok(ApiResponse.success(
            commentService.getCommentsByTask(taskId)));
    }

    /** Team member resolves a comment */
    @PutMapping("/{id}/resolve")
    public ResponseEntity<ApiResponse<Response>> resolveComment(
            @PathVariable String id,
            @AuthenticationPrincipal UserDetails user) {
        return ResponseEntity.ok(ApiResponse.success("Comment resolved",
            commentService.resolveComment(id, user.getUsername())));
    }
}
