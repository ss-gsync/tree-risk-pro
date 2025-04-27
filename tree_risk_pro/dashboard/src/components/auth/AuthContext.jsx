// src/components/auth/AuthContext.jsx
/**
 * Authentication context provider for Tree Risk Pro dashboard
 * 
 * Manages authentication state and provides login/logout functionality
 * Uses localStorage for token persistence and validates token on load
 */
import React, { createContext, useState, useContext, useEffect } from 'react';

// Create authentication context
const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  // Authentication state
  const [authToken, setAuthToken] = useState(localStorage.getItem('auth') || null);
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem('auth'));
  const [isValidating, setIsValidating] = useState(true);

  // Validate authentication token on mount or when token changes
  useEffect(() => {
    // Track if component is mounted to prevent state updates after unmount
    let isMounted = true;
    
    const validateToken = async () => {
      if (!isMounted) return;
      setIsValidating(true);
      
      if (!authToken) {
        setIsAuthenticated(false);
        setIsValidating(false);
        return;
      }

      try {
        // Security: Add timeout to prevent hanging authentication
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        
        // Validate token with lightweight API request
        const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:5000'}/api/properties`, {
          headers: {
            'Authorization': authToken
          },
          signal: controller.signal,
          // Don't send cookies with the request for added security
          credentials: 'omit'
        });

        clearTimeout(timeoutId);
        
        if (!isMounted) return;

        if (response.ok) {
          setIsAuthenticated(true);
          
          // Security: Periodically re-validate token (token refresh)
          // This helps detect revoked tokens and session timeouts
          const refreshTimer = setTimeout(validateToken, 15 * 60 * 1000); // 15 minutes
          return () => clearTimeout(refreshTimer);
        } else {
          // Security: Clear invalid token from all storage
          setAuthToken(null);
          localStorage.removeItem('auth');
          sessionStorage.removeItem('auth');
          setIsAuthenticated(false);
          
          // Security: For 403 (forbidden) errors, log the user out completely
          if (response.status === 403) {
            console.warn('Access forbidden - user permissions may have changed');
            window.location.reload();
          }
        }
      } catch (error) {
        if (!isMounted) return;
        
        if (error.name === 'AbortError') {
          console.warn('Auth validation timed out');
        } else {
          console.error('Error validating auth token:', error);
        }
        
        // Don't automatically clear token on network errors
        // This prevents logging users out during temporary connectivity issues
        setIsAuthenticated(false);
      } finally {
        if (isMounted) {
          setIsValidating(false);
        }
      }
    };

    validateToken();
    
    // Clean up function to prevent memory leaks and state updates after unmount
    return () => {
      isMounted = false;
    };
  }, [authToken]);

  // Login - store token and update state with security measures
  const login = (token) => {
    if (!token) {
      console.error('Attempted login with empty token');
      return;
    }
    
    try {
      // Set auth state
      setAuthToken(token);
      
      // Store in localStorage for persistence across page reloads
      localStorage.setItem('auth', token);
      
      // Store login timestamp for session expiry calculations
      localStorage.setItem('auth_timestamp', Date.now().toString());
      
      // Update authenticated state
      setIsAuthenticated(true);
      
      // Security: Log login event (in a real app, this could go to an audit log)
      console.info('User authenticated successfully');
    } catch (error) {
      console.error('Error during login:', error);
    }
  };

  // Secure logout - properly clean up all auth data
  const logout = () => {
    try {
      // Security: Clear all sensitive data
      setAuthToken(null);
      setIsAuthenticated(false);
      
      // Remove auth data from all storage locations
      localStorage.removeItem('auth');
      localStorage.removeItem('auth_timestamp');
      sessionStorage.removeItem('auth');
      
      // Clear any data caches that might contain sensitive information
      if (caches && caches.delete) {
        caches.delete('api-data').catch(err => console.error('Error clearing cache:', err));
      }
      
      // Security: Log logout event
      console.info('User logged out');
      
      // Force a complete refresh to clear memory and reload all components
      window.location.href = window.location.pathname;
    } catch (error) {
      console.error('Error during logout:', error);
      // Fallback - force reload even if error occurs to ensure logout
      window.location.reload();
    }
  };

  // Add session security check for token expiry
  const checkSessionExpiry = () => {
    try {
      const loginTime = parseInt(localStorage.getItem('auth_timestamp') || '0', 10);
      if (!loginTime) return false;
      
      // Check if session has expired (e.g., after 8 hours)
      const SESSION_DURATION = 8 * 60 * 60 * 1000; // 8 hours in milliseconds
      const isExpired = Date.now() - loginTime > SESSION_DURATION;
      
      if (isExpired && authToken) {
        console.warn('Session expired, logging out');
        logout();
        return true;
      }
      return false;
    } catch (error) {
      console.error('Error checking session expiry:', error);
      return false;
    }
  };
  
  // Check for session expiry on component mount and periodically
  useEffect(() => {
    // Initial check
    checkSessionExpiry();
    
    // Periodically check for session expiry
    const intervalId = setInterval(checkSessionExpiry, 5 * 60 * 1000); // Check every 5 minutes
    
    return () => clearInterval(intervalId);
  }, []);
  
  // Context value with enhanced security functions
  const value = {
    authToken,
    isAuthenticated,
    isValidating,
    login,
    logout,
    // Additional security functions
    checkSessionExpiry,
    // Function to get remaining session time
    getSessionTimeRemaining: () => {
      try {
        const loginTime = parseInt(localStorage.getItem('auth_timestamp') || '0', 10);
        if (!loginTime) return 0;
        
        const SESSION_DURATION = 8 * 60 * 60 * 1000; // 8 hours
        const remaining = Math.max(0, SESSION_DURATION - (Date.now() - loginTime));
        return remaining;
      } catch (error) {
        return 0;
      }
    }
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

// Custom hook for using auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};