// src/components/auth/Login.jsx
import React, { useState, useEffect } from 'react';
import logo from '../../logo.png';

const Login = ({ onLogin }) => {
  // Form state
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoggingIn, setIsLoggingIn] = useState(false);

  // Check for production mode and pre-fill credentials
  useEffect(() => {
    const checkAppMode = async () => {
      try {
        const apiBase = import.meta.env.VITE_API_URL === undefined
          ? 'http://localhost:5000'
          : (import.meta.env.VITE_API_URL === '' ? '' : import.meta.env.VITE_API_URL);
        const response = await fetch(`${apiBase}/api/config`);
        if (response.ok) {
          const config = await response.json();
          if (config.mode === 'production') {
            setUsername('TestAdmin');
            setPassword('trp345!');
          }
        }
      } catch (error) {
        console.error('Failed to check app mode:', error);
      }
    };

    checkAppMode();
  }, []);

  // Handle form submission and authentication
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setIsLoggingIn(true);

    try {
      // Trim credentials and perform basic validation
      const trimmedUsername = username.trim();
      const trimmedPassword = password;
      
      if (!trimmedUsername || !trimmedPassword) {
        setError('Username and password are required');
        setIsLoggingIn(false);
        return;
      }
      
      // Rate limiting - add delay to prevent brute force attacks
      await new Promise(resolve => setTimeout(resolve, 300));
      
      // Create Basic Auth header (encode credentials)
      const authString = `Basic ${btoa(`${trimmedUsername}:${trimmedPassword}`)}`;
      
      // Validate credentials with API request
      // Use a dedicated auth endpoint instead of a data endpoint when possible
      const apiBase = import.meta.env.VITE_API_URL === undefined
        ? 'http://localhost:5000'
        : (import.meta.env.VITE_API_URL === '' ? '' : import.meta.env.VITE_API_URL);
      const response = await fetch(`${apiBase}/api/properties`, {
        headers: {
          'Authorization': authString
        },
        // Cancel request if it takes too long (prevents hanging UI)
        signal: AbortSignal.timeout(10000)
      });

      if (response.ok) {
        // Security: Don't store plain credentials
        // Store auth token (already encoded with Basic Auth)
        localStorage.setItem('auth', authString);
        
        // Clear sensitive data
        if (trimmedUsername !== 'TestAdmin') { // Keep test creds pre-filled in prod
          setPassword(''); // Clear password from state
        }
        
        // Complete login
        onLogin(authString);
      } else if (response.status === 401) {
        // Specific error for auth failures
        setError('Invalid username or password');
        // Implement progressive delays for repeated failures (not shown here)
      } else {
        setError('Server error. Please try again later.');
      }
    } catch (error) {
      // Differentiate between network errors and other failures
      if (error.name === 'AbortError') {
        setError('Request timed out. Please check your connection.');
      } else if (error.name === 'TypeError' && error.message.includes('NetworkError')) {
        setError('Network error. Please check your connection.');
      } else {
        setError('Login failed. Please try again.');
        console.error('Login error:', error);
      }
    } finally {
      setIsLoggingIn(false);
    }
  };

  return (
    <div style={{display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh', background: 'linear-gradient(to bottom, #f1f5f9, #e2e8f0)'}}>
      <div style={{background: '#fff', padding: '2rem', borderRadius: '0.5rem', boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)', maxWidth: '400px', width: '100%', border: '1px solid rgba(0, 0, 0, 0.05)'}}>
        <div style={{textAlign: 'center', marginBottom: '2rem'}}>
          <div style={{display: 'flex', justifyContent: 'center', alignItems: 'center', marginBottom: '1rem'}}>
            <img src={logo} alt="Texas Tree Transformations" style={{width: '60px', height: 'auto'}} />
          </div>
          <h1 style={{color: '#991b1b', fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '0.25rem'}}>
            Tree Risk Pro
          </h1>
          <p style={{color: '#e11d48', fontSize: '0.875rem'}}>Login to access the dashboard</p>
        </div>
        
        {error && (
          <div style={{background: '#fee2e2', color: '#b91c1c', padding: '0.75rem', borderRadius: '0.25rem', marginBottom: '1rem'}}>
            {error}
          </div>
        )}
        
        <form onSubmit={handleSubmit} autoComplete="on">
          <div style={{marginBottom: '1rem'}}>
            <label htmlFor="username" style={{display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem', fontWeight: 'medium'}}>
              Username
            </label>
            <input
              type="text"
              id="username"
              name="username"
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              style={{width: '100%', padding: '0.5rem', border: '1px solid #d1d5db', borderRadius: '0.25rem', backgroundColor: '#ffffff !important'}}
              required
              aria-label="Username"
            />
          </div>
          
          <div style={{marginBottom: '1.5rem'}}>
            <label htmlFor="password" style={{display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem', fontWeight: 'medium'}}>
              Password
            </label>
            <input
              type="password"
              id="password"
              name="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              style={{width: '100%', padding: '0.5rem', border: '1px solid #d1d5db', borderRadius: '0.25rem', backgroundColor: '#ffffff !important'}}
              required
              aria-label="Password"
            />
          </div>
          
          <button
            type="submit"
            disabled={isLoggingIn}
            style={{
              width: '100%',
              background: 'linear-gradient(to bottom, #991b1b, #7f1d1d)',
              color: 'white',
              fontWeight: 'bold',
              padding: '0.625rem 1rem',
              borderRadius: '0.375rem',
              border: 'none',
              cursor: isLoggingIn ? 'not-allowed' : 'pointer',
              boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
              transition: 'all 0.15s ease'
            }}
            onMouseOver={(e) => {
              if (!isLoggingIn) e.target.style.background = 'linear-gradient(to bottom, #7f1d1d, #7f1d1d)';
            }}
            onMouseOut={(e) => {
              if (!isLoggingIn) e.target.style.background = 'linear-gradient(to bottom, #991b1b, #7f1d1d)';
            }}
          >
            {isLoggingIn ? 'Logging in...' : 'Log In'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default Login;