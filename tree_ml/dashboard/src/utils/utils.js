/**
 * Utility functions for the application
 */

/**
 * Generate a detection job ID with consistent formatting
 * This function serves as the SINGLE SOURCE OF TRUTH for job ID generation
 * All components should use this function to ensure consistency
 * 
 * @returns {string} A job ID in the format "detection_[timestamp]"
 */
export function generateDetectionJobId() {
  const timestamp = Date.now(); // Full timestamp with milliseconds
  const jobId = `detection_${timestamp}`;
  console.log(`Utils: Generated job ID: ${jobId}`);
  return jobId;
}

// Global variable to store the current job ID
// This serves as an additional safety mechanism
let currentJobId = null;

/**
 * Set the current job ID for the application
 * This function is called when a detection is started
 * 
 * @param {string} jobId The job ID to set
 * @returns {string} The job ID that was set
 */
export function setCurrentJobId(jobId) {
  currentJobId = jobId;
  // Store in window for cross-component access
  window.currentDetectionJobId = jobId;
  console.log(`Utils: Set current job ID: ${jobId}`);
  return jobId;
}

/**
 * Get the current job ID for the application
 * This function is called when a component needs to know the current job ID
 * 
 * @returns {string|null} The current job ID or null if not set
 */
export function getCurrentJobId() {
  // Prefer window variable if available (for cross-tab consistency)
  return window.currentDetectionJobId || currentJobId;
}