package com.teamtrack.auth.repository;

import com.teamtrack.auth.model.User;
import com.teamtrack.util.Role;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * User Repository
 *
 * @Repository            - Marks as a Spring Data repository bean; enables exception translation
 * MongoRepository<T, ID> - Provides CRUD methods: save, findById, findAll, delete, count, etc.
 *
 * Spring Data MongoDB generates implementations for method names following naming conventions:
 * findBy + FieldName = WHERE field = value
 * findBy + Field + And + Field = WHERE field1 = v1 AND field2 = v2
 *
 * @Query - Custom MongoDB JSON query when method name convention is insufficient
 */
@Repository
public interface UserRepository extends MongoRepository<User, String> {

    Optional<User> findByEmail(String email);

    boolean existsByEmail(String email);

    List<User> findByRole(Role role);

    List<User> findByTeamId(String teamId);

    List<User> findByTeamIdAndRole(String teamId, Role role);

    @Query("{ 'is_active': true, 'role': ?0 }")
    List<User> findActiveUsersByRole(String role);

    long countByTeamId(String teamId);
}
