// src/components/ui/card.jsx

import React from 'react';

const Card = ({ className = '', children, ...props }) => (
  <div className={`bg-white shadow-sm rounded-lg overflow-hidden ${className}`} {...props}>
    {children}
  </div>
);

const CardHeader = ({ className = '', children, ...props }) => (
  <div className={`p-4 ${className}`} {...props}>
    {children}
  </div>
);

const CardTitle = ({ className = '', children, ...props }) => (
  <h3 className={`text-lg font-semibold ${className}`} {...props}>
    {children}
  </h3>
);

const CardDescription = ({ className = '', children, ...props }) => (
  <p className={`text-sm text-gray-500 ${className}`} {...props}>
    {children}
  </p>
);

const CardContent = ({ className = '', children, ...props }) => (
  <div className={`p-4 pt-0 ${className}`} {...props}>
    {children}
  </div>
);

const CardFooter = ({ className = '', children, ...props }) => (
  <div className={`p-4 pt-0 ${className}`} {...props}>
    {children}
  </div>
);

// Special card for Gemini AI features
const GeminiCard = ({ active = false, className = '', children, ...props }) => (
  <div 
    className={`shadow-sm rounded-lg overflow-hidden transition-colors duration-300 ${
      active ? 'bg-blue-50 border border-blue-300' : 'bg-gray-50 border border-gray-200'
    } ${className}`} 
    {...props}
  >
    {children}
  </div>
);

export { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter, GeminiCard };
