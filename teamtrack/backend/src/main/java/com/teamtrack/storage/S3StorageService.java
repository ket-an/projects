package com.teamtrack.storage;

import com.teamtrack.config.AppProperties;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.*;
import software.amazon.awssdk.services.s3.presigner.S3Presigner;
import software.amazon.awssdk.services.s3.presigner.model.*;

import java.io.InputStream;
import java.time.Duration;
import java.util.UUID;

/**
 * S3 Storage Service
 *
 * @Service - Spring service layer stereotype
 * Handles: pre-signed PUT URLs for upload, pre-signed GET URLs for download, object deletion
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class S3StorageService {

    private final S3Client s3Client;
    private final S3Presigner s3Presigner;
    private final AppProperties appProperties;

    /**
     * Generates a pre-signed PUT URL so the client can upload directly to S3.
     * File bytes never pass through the Spring Boot backend.
     */
    public String generateUploadUrl(String folder, String originalFileName) {
        String s3Key = folder + "/" + UUID.randomUUID() + "-" + originalFileName;
        String bucket = appProperties.getAws().getS3().getBucketName();
        long expiry = appProperties.getAws().getS3().getPresignedUrlExpiry();

        PutObjectPresignRequest presignRequest = PutObjectPresignRequest.builder()
            .signatureDuration(Duration.ofSeconds(expiry))
            .putObjectRequest(PutObjectRequest.builder()
                .bucket(bucket)
                .key(s3Key)
                .build())
            .build();

        return s3Presigner.presignPutObject(presignRequest).url().toString();
    }

    /**
     * Generates a pre-signed GET URL for a stored object (e.g. report download).
     */
    public String generateDownloadUrl(String s3Key) {
        String bucket = appProperties.getAws().getS3().getBucketName();
        long expiry = appProperties.getAws().getS3().getReportUrlExpiry();

        GetObjectPresignRequest presignRequest = GetObjectPresignRequest.builder()
            .signatureDuration(Duration.ofSeconds(expiry))
            .getObjectRequest(GetObjectRequest.builder()
                .bucket(bucket)
                .key(s3Key)
                .build())
            .build();

        return s3Presigner.presignGetObject(presignRequest).url().toString();
    }

    /**
     * Uploads bytes directly from backend (used for generated reports).
     */
    public String uploadBytes(byte[] data, String s3Key, String contentType) {
        String bucket = appProperties.getAws().getS3().getBucketName();

        s3Client.putObject(
            PutObjectRequest.builder()
                .bucket(bucket)
                .key(s3Key)
                .contentType(contentType)
                .contentLength((long) data.length)
                .build(),
            RequestBody.fromBytes(data));

        log.info("Uploaded {} bytes to s3://{}/{}", data.length, bucket, s3Key);
        return s3Key;
    }

    public void deleteObject(String s3Key) {
        try {
            s3Client.deleteObject(DeleteObjectRequest.builder()
                .bucket(appProperties.getAws().getS3().getBucketName())
                .key(s3Key)
                .build());
            log.info("Deleted S3 object: {}", s3Key);
        } catch (S3Exception e) {
            log.warn("Failed to delete S3 object {}: {}", s3Key, e.getMessage());
        }
    }
}
