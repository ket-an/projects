package com.teamtrack.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.presigner.S3Presigner;

/**
 * AWS S3 Configuration
 *
 * @Configuration  - Marks as Spring config class
 * @Value          - Injects values from application.yml / environment variables
 *                   Supports SpEL and default values via :
 * @Profile        - Activates bean only for specific Spring profiles
 */
@Configuration
public class S3Config {

    @Value("${app.aws.region}")
    private String region;

    @Value("${app.aws.access-key}")
    private String accessKey;

    @Value("${app.aws.secret-key}")
    private String secretKey;

    /**
     * S3Client for bucket operations (upload, delete, list)
     * Uses StaticCredentialsProvider for local/dev; in production use DefaultCredentialsProvider
     * which automatically picks up IAM role credentials on EKS via IRSA
     */
    @Bean
    public S3Client s3Client() {
        if (accessKey.startsWith("local") || accessKey.isEmpty()) {
            // Local dev: use default credential chain (env vars, ~/.aws/credentials, EC2/EKS IRSA)
            return S3Client.builder()
                .region(Region.of(region))
                .credentialsProvider(DefaultCredentialsProvider.create())
                .build();
        }
        return S3Client.builder()
            .region(Region.of(region))
            .credentialsProvider(StaticCredentialsProvider.create(
                AwsBasicCredentials.create(accessKey, secretKey)))
            .build();
    }

    /**
     * S3Presigner for generating pre-signed URLs (PUT for upload, GET for download)
     */
    @Bean
    public S3Presigner s3Presigner() {
        if (accessKey.startsWith("local") || accessKey.isEmpty()) {
            return S3Presigner.builder()
                .region(Region.of(region))
                .credentialsProvider(DefaultCredentialsProvider.create())
                .build();
        }
        return S3Presigner.builder()
            .region(Region.of(region))
            .credentialsProvider(StaticCredentialsProvider.create(
                AwsBasicCredentials.create(accessKey, secretKey)))
            .build();
    }
}
