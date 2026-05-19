package com.teamtrack.config;

import com.mongodb.client.MongoClient;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.domain.Sort;
import org.springframework.data.mongodb.MongoDatabaseFactory;
import org.springframework.data.mongodb.MongoTransactionManager;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.index.CompoundIndexDefinition;
import org.springframework.data.mongodb.core.index.Index;
import org.springframework.data.mongodb.core.index.IndexOperations;
import org.springframework.data.document.Document;

/**
 * MongoDB Configuration
 *
 * @Configuration  - Spring configuration class
 * @Slf4j          - Lombok: injects 'log' SLF4J logger field
 * @PostConstruct  - Method runs once after dependency injection completes
 */
@Configuration
@RequiredArgsConstructor
@Slf4j
public class MongoConfig {

    private final MongoTemplate mongoTemplate;

    /**
     * @PostConstruct - Runs after bean initialization; used here to create MongoDB indexes
     * programmatically (in addition to @Indexed annotations on domain classes)
     */
    @PostConstruct
    public void createIndexes() {
        log.info("Creating MongoDB indexes...");

        // Tasks indexes
        IndexOperations taskOps = mongoTemplate.indexOps("tasks");
        taskOps.ensureIndex(new CompoundIndexDefinition(
            new org.bson.Document("weekId", 1).append("userId", 1)));
        taskOps.ensureIndex(new Index().on("userId", Sort.Direction.ASC)
            .on("createdAt", Sort.Direction.DESC));
        taskOps.ensureIndex(new Index().on("status", Sort.Direction.ASC));

        // Weeks indexes
        IndexOperations weekOps = mongoTemplate.indexOps("weeks");
        weekOps.ensureIndex(new CompoundIndexDefinition(
            new org.bson.Document("userId", 1).append("status", 1)));
        weekOps.ensureIndex(new Index().on("startDate", Sort.Direction.ASC)
            .on("endDate", Sort.Direction.ASC));

        // Comments indexes
        IndexOperations commentOps = mongoTemplate.indexOps("comments");
        commentOps.ensureIndex(new CompoundIndexDefinition(
            new org.bson.Document("taskId", 1).append("createdAt", 1)));
        commentOps.ensureIndex(new CompoundIndexDefinition(
            new org.bson.Document("taskId", 1).append("resolved", 1)));

        // Users index
        IndexOperations userOps = mongoTemplate.indexOps("users");
        userOps.ensureIndex(new Index().on("email", Sort.Direction.ASC).unique());

        log.info("MongoDB indexes created successfully");
    }

    /**
     * @Bean - MongoTransactionManager enables multi-document ACID transactions
     * (requires MongoDB replica set)
     */
    @Bean
    public MongoTransactionManager transactionManager(MongoDatabaseFactory dbFactory) {
        return new MongoTransactionManager(dbFactory);
    }
}
