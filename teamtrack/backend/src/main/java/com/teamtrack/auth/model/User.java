package com.teamtrack.auth.model;

import com.teamtrack.util.Role;
import lombok.*;
import org.springframework.data.annotation.*;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

import java.time.LocalDateTime;
import java.util.List;

/**
 * User MongoDB Document
 *
 * @Document         - Maps this class to a MongoDB collection named "users"
 * @Id               - Marks the primary key field (maps to MongoDB _id)
 * @Indexed(unique)  - Creates a unique index on the email field
 * @CreatedDate      - Automatically set by @EnableMongoAuditing when document is saved
 * @LastModifiedDate - Automatically updated on every save by MongoDB auditing
 * @Field            - Overrides the MongoDB field name (snake_case in DB, camelCase in Java)
 * @Data             - Lombok: generates all boilerplate (getters, setters, equals, hashCode, toString)
 * @Builder          - Lombok: generates builder pattern
 * @NoArgsConstructor / @AllArgsConstructor - Lombok: generates constructors
 */
@Document(collection = "users")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class User {

    @Id
    private String id;

    @Field("name")
    private String name;

    @Indexed(unique = true)
    @Field("email")
    private String email;

    @Field("password_hash")
    private String passwordHash;

    @Field("role")
    private Role role;

    @Field("team_id")
    private String teamId;

    @Field("department")
    private String department;

    @Field("profile_image_url")
    private String profileImageUrl;

    @Field("is_active")
    @Builder.Default
    private boolean active = true;

    @CreatedDate
    @Field("created_at")
    private LocalDateTime createdAt;

    @LastModifiedDate
    @Field("updated_at")
    private LocalDateTime updatedAt;
}
