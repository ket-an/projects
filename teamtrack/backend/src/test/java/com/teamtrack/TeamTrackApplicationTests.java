package com.teamtrack;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

/**
 * Application context load test
 *
 * @SpringBootTest     - Loads the full application context for integration testing
 * @ActiveProfiles     - Activates "test" Spring profile
 * @TestPropertySource - Overrides properties for tests
 */
@SpringBootTest
@ActiveProfiles("test")
@TestPropertySource(properties = {
    "spring.data.mongodb.uri=mongodb://localhost:27017/teamtrack_test",
    "app.aws.access-key=test-key",
    "app.aws.secret-key=test-secret",
    "app.aws.s3.bucket-name=test-bucket"
})
class TeamTrackApplicationTests {

    @Test
    void contextLoads() {
        // Verifies Spring context starts without errors
    }
}
