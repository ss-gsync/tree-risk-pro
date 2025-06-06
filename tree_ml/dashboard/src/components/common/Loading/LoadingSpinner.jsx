// src/components/common/Loading/LoadingSpinner.jsx

import React from 'react';

const LoadingSpinner = () => {
  return (
    <div className="flex items-center justify-center h-screen">
      <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-green-600"></div>
    </div>
  );
};

export default LoadingSpinner;