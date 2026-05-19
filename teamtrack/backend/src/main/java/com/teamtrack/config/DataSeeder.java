package com.teamtrack.config;

import com.teamtrack.auth.model.User;
import com.teamtrack.auth.repository.UserRepository;
import com.teamtrack.util.Role;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.annotation.Profile;
import org.springframework.context.event.EventListener;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

/**
 * Data Seeder — seeds default users for local development
 *
 * @Profile       - Only activates this bean when Spring profile is "dev" or "default".
 *                  In prod (--spring.profiles.active=prod) this bean is never created.
 * @EventListener - Listens for ApplicationReadyEvent (fired after full context startup).
 *                  Preferred over @PostConstruct when you need the full context ready.
 */
@Component
@Profile({"dev", "default"})
@RequiredArgsConstructor
@Slf4j
public class DataSeeder {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    @EventListener(ApplicationReadyEvent.class)
    public void seedData() {
        if (userRepository.count() > 0) {
            log.info("Database already seeded — skipping");
            return;
        }

        log.info("Seeding default users...");

        // Manager
        userRepository.save(User.builder()
            .name("Arjun Sharma")
            .email("manager@teamtrack.com")
            .passwordHash(passwordEncoder.encode("Password@123"))
            .role(Role.MANAGER)
            .teamId("TEAM-ALPHA")
            .department("Engineering")
            .active(true)
            .build());

        // Team Members
        String[] names = {"Ketan Patil", "Priya Nair", "Rahul Gupta", "Swarnima Singh"};
        String[] emails = {"ketan@teamtrack.com", "priya@teamtrack.com",
                           "rahul@teamtrack.com", "swarnima@teamtrack.com"};

        for (int i = 0; i < names.length; i++) {
            userRepository.save(User.builder()
                .name(names[i])
                .email(emails[i])
                .passwordHash(passwordEncoder.encode("Password@123"))
                .role(Role.TEAM_MEMBER)
                .teamId("TEAM-ALPHA")
                .department("Engineering")
                .active(true)
                .build());
        }

        log.info("Seeded {} users. Login: manager@teamtrack.com / Password@123",
            userRepository.count());
    }
}
