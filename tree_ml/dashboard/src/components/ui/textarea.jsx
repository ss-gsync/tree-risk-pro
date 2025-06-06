// src/components/ui/textarea.jsx

import React from 'react';

const Textarea = React.forwardRef(({
  className = '',
  ...props
}, ref) => {
  return (
    <textarea
      className={`flex min-h-[60px] w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm
        placeholder:text-gray-400
        focus:outline-none focus:ring-1 focus:ring-emerald-400 focus:border-emerald-400
        disabled:cursor-not-allowed disabled:opacity-50
        ${className}`}
      ref={ref}
      {...props}
    />
  );
});

Textarea.displayName = 'Textarea';

export { Textarea };
