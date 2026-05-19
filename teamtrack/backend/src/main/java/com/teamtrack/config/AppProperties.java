package com.teamtrack.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * Strongly-typed application properties
 *
 * @ConfigurationProperties - Binds all properties with prefix "app" from application.yml
 * @Component              - Registers as Spring bean so it can be @Autowired anywhere
 * @Data                   - Lombok: generates getters, setters, toString, equals, hashCode
 */
@ConfigurationProperties(prefix = "app")
@Component
@Data
public class AppProperties {

    private Jwt jwt = new Jwt();
    private Aws aws = new Aws();
    private Pagination pagination = new Pagination();

    @Data
    public static class Jwt {
        private String secret;
        private long expiration;
        private long refreshExpiration;
    }

    @Data
    public static class Aws {
        private String region;
        private S3 s3 = new S3();
        private String accessKey;
        private String secretKey;

        @Data
        public static class S3 {
            private String bucketName;
            private long presignedUrlExpiry;
            private long reportUrlExpiry;
        }
    }

    @Data
    public static class Pagination {
        private int defaultPageSize;
        private int maxPageSize;
    }
}
