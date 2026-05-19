package com.teamtrack.auth.controller;

import com.teamtrack.auth.dto.AuthDto.UserDto;
import com.teamtrack.auth.model.User;
import com.teamtrack.auth.repository.UserRepository;
import com.teamtrack.auth.service.AuthService;
import com.teamtrack.exception.ResourceNotFoundException;
import com.teamtrack.util.ApiResponse;
import com.teamtrack.util.Role;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Manager-only user management endpoints.
 * All methods protected by @PreAuthorize at class and method level.
 */
@RestController
@RequestMapping("/manager/users")
@PreAuthorize("hasRole('MANAGER')")
@RequiredArgsConstructor
public class UserManagementController {

    private final UserRepository userRepository;

    @GetMapping
    public ApiResponse<List<UserDto>> getAllTeamMembers(
            @RequestParam(required = false) String teamId) {
        List<User> users = (teamId != null)
            ? userRepository.findByTeamIdAndRole(teamId, Role.TEAM_MEMBER)
            : userRepository.findByRole(Role.TEAM_MEMBER);

        List<UserDto> dtos = users.stream()
            .map(AuthService::toUserDto)
            .collect(Collectors.toList());
        return ApiResponse.success(dtos);
    }

    @GetMapping("/{id}")
    public ApiResponse<UserDto> getMemberById(@PathVariable String id) {
        User user = userRepository.findById(id)
            .orElseThrow(() -> new ResourceNotFoundException("User", "id", id));
        return ApiResponse.success(AuthService.toUserDto(user));
    }

    @PutMapping("/{id}/deactivate")
    public ApiResponse<Void> deactivateUser(@PathVariable String id) {
        User user = userRepository.findById(id)
            .orElseThrow(() -> new ResourceNotFoundException("User", "id", id));
        user.setActive(false);
        userRepository.save(user);
        return ApiResponse.success("User deactivated", null);
    }
}
