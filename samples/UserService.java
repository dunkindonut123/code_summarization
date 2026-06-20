package com.example.service;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Manages user accounts: lookup, registration, and password updates.
 */
public class UserService {

    private final Map<String, String> usersByEmail = new HashMap<>();

    public Optional<String> findUserIdByEmail(String email) {
        if (email == null || email.isBlank()) {
            return Optional.empty();
        }
        String normalized = email.trim().toLowerCase();
        return Optional.ofNullable(usersByEmail.get(normalized));
    }

    public boolean registerUser(String email, String userId) {
        if (email == null || userId == null || email.isBlank() || userId.isBlank()) {
            return false;
        }
        String normalized = email.trim().toLowerCase();
        if (usersByEmail.containsKey(normalized)) {
            return false;
        }
        usersByEmail.put(normalized, userId);
        return true;
    }

    public boolean updatePassword(String email, String newPasswordHash) {
        if (email == null || newPasswordHash == null || newPasswordHash.isBlank()) {
            return false;
        }
        String normalized = email.trim().toLowerCase();
        if (!usersByEmail.containsKey(normalized)) {
            return false;
        }
        // In a real app this would persist to a credential store.
        return true;
    }
}
