// src/utils/toast.js
// Simple toast notification utilities

// This is a simple implementation. In a production app,
// you might want to use a library like react-toastify or react-hot-toast.

const TOAST_DURATION = 3000; // 3 seconds

// Create and append a toast element to the DOM
function createToast(message, type = 'info') {
  // Check if the toast container exists, create it if not
  let toastContainer = document.getElementById('toast-container');
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.id = 'toast-container';
    toastContainer.style.position = 'fixed';
    toastContainer.style.top = '1rem';
    toastContainer.style.right = '1rem';
    toastContainer.style.zIndex = '9999';
    document.body.appendChild(toastContainer);
  }

  // Create the toast element
  const toast = document.createElement('div');
  toast.style.minWidth = '250px';
  toast.style.margin = '0.5rem';
  toast.style.padding = '1rem 1.5rem';
  toast.style.borderRadius = '0.375rem';
  toast.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)';
  toast.style.opacity = '0';
  toast.style.transition = 'opacity 0.3s ease-in-out';
  toast.textContent = message;

  // Set styles based on toast type
  switch (type) {
    case 'success':
      toast.style.backgroundColor = '#10B981'; // Green
      toast.style.color = 'white';
      break;
    case 'error':
      toast.style.backgroundColor = '#EF4444'; // Red
      toast.style.color = 'white';
      break;
    case 'warning':
      toast.style.backgroundColor = '#F59E0B'; // Amber
      toast.style.color = 'white';
      break;
    case 'info':
    default:
      toast.style.backgroundColor = '#3B82F6'; // Blue
      toast.style.color = 'white';
      break;
  }

  // Add to container
  toastContainer.appendChild(toast);

  // Fade in
  setTimeout(() => {
    toast.style.opacity = '1';
  }, 10);

  // Fade out and remove after duration
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => {
      toastContainer.removeChild(toast);
      // Remove container if empty
      if (toastContainer.children.length === 0) {
        document.body.removeChild(toastContainer);
      }
    }, 300);
  }, TOAST_DURATION);
}

// Export various toast types
export function info(message) {
  createToast(message, 'info');
}

export function success(message) {
  createToast(message, 'success');
}

export function error(message) {
  createToast(message, 'error');
}

export function warning(message) {
  createToast(message, 'warning');
}

// Alias for common use case
export function save(message = 'Saved successfully') {
  success(message);
}