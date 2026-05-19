package com.teamtrack;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.data.mongodb.config.EnableMongoAuditing;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * TeamTrack Application Entry Point
 *
 * @SpringBootApplication          - Combines @Configuration + @EnableAutoConfiguration + @ComponentScan
 * @EnableMongoAuditing            - Enables @CreatedDate, @LastModifiedDate on MongoDB documents
 * @EnableAsync                    - Enables @Async method execution (used in EmailService)
 * @EnableScheduling               - Enables @Scheduled tasks (e.g. report cleanup)
 * @EnableCaching                  - Enables Spring Cache abstraction (@Cacheable, @CacheEvict)
 */
@SpringBootApplication
@EnableMongoAuditing
@EnableAsync
@EnableScheduling
@EnableCaching
public class TeamTrackApplication {

    public static void main(String[] args) {
        SpringApplication.run(TeamTrackApplication.class, args);
    }
}
