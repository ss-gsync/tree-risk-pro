// src/components/ui/input.jsx

import React from 'react';

const Input = React.forwardRef(({
  className = '',
  type = 'text',
  ...props
}, ref) => {
  return (
    <input
      type={type}
      className={`flex h-9 w-full rounded-md border border-gray-300 bg-white px-3 py-1 text-sm shadow-sm transition-colors
        file:border-0 file:bg-transparent file:text-sm file:font-medium
        placeholder:text-gray-400
        focus:outline-none focus:ring-1 focus:ring-emerald-400 focus:border-emerald-400
        disabled:cursor-not-allowed disabled:opacity-50
        ${className}`}
      ref={ref}
      {...props}
    />
  );
});

Input.displayName = 'Input';

export { Input };
