// src/components/ui/button.jsx

import React from 'react';

const Button = ({
  children,
  className = '',
  variant = 'default',
  size = 'default',
  disabled = false,
  type = 'button',
  ...props
}) => {
  // Base styles
  let buttonClasses = 'inline-flex items-center justify-center rounded-md font-medium transition-colors';
  
  // Variant styles
  if (variant === 'default') {
    buttonClasses += ' bg-emerald-500 text-white hover:bg-emerald-600 active:bg-emerald-700';
  } else if (variant === 'outline') {
    buttonClasses += ' border border-gray-300 bg-transparent text-gray-700 hover:bg-gray-50';
  } else if (variant === 'ghost') {
    buttonClasses += ' bg-transparent text-gray-700 hover:bg-gray-100';
  }
  
  // Size styles
  if (size === 'default') {
    buttonClasses += ' h-10 py-2 px-4 text-sm';
  } else if (size === 'sm') {
    buttonClasses += ' h-8 px-3 text-xs';
  } else if (size === 'lg') {
    buttonClasses += ' h-12 px-6 text-base';
  }
  
  // Disabled styles
  if (disabled) {
    buttonClasses += ' opacity-50 cursor-not-allowed';
  }
  
  return (
    <button
      className={`${buttonClasses} ${className}`}
      disabled={disabled}
      type={type}
      {...props}
    >
      {children}
    </button>
  );
};

export { Button };
