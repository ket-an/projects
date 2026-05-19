package com.teamtrack.auth.service;

import com.teamtrack.auth.model.User;
import com.teamtrack.auth.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.*;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Custom UserDetailsService
 *
 * @Service - Spring stereotype for service-layer beans
 * UserDetailsService - Spring Security interface; loadUserByUsername() is called during authentication
 */
@Service
@RequiredArgsConstructor
public class UserDetailsServiceImpl implements UserDetailsService {

    private final UserRepository userRepository;

    /**
     * Called by Spring Security's DaoAuthenticationProvider to load user for authentication.
     * Maps our User domain object to Spring Security's UserDetails.
     * Role is prefixed with "ROLE_" so "MANAGER" becomes "ROLE_MANAGER".
     */
    @Override
    public UserDetails loadUserByUsername(String email) throws UsernameNotFoundException {
        User user = userRepository.findByEmail(email)
            .orElseThrow(() -> new UsernameNotFoundException(
                "User not found with email: " + email));

        return new org.springframework.security.core.userdetails.User(
            user.getEmail(),
            user.getPasswordHash(),
            user.isActive(),
            true, true, true,
            List.of(new SimpleGrantedAuthority("ROLE_" + user.getRole().name()))
        );
    }
}
