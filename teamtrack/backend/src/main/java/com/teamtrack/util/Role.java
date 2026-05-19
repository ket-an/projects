package com.teamtrack.util;

/**
 * User role enum used for Spring Security RBAC.
 * Spring Security prefixes roles with "ROLE_" internally when using hasRole().
 * So MANAGER stored here becomes ROLE_MANAGER in the security context.
 */
public enum Role {
    TEAM_MEMBER,
    MANAGER
}
