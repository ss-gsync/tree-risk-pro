// src/components/common/Layout/MainContent/index.jsx

import React from 'react';

/**
 * MainContent component - handles the main content area of the application
 * 
 * This component is deliberately simple with no dynamic padding calculations.
 * Instead, it provides a clean full-width/height container for child components,
 * and lets each view handle its own sizing and positioning relative to sidebars.
 * 
 * This approach avoids layout issues with invisible padding or nested containers.
 */
const MainContent = ({ children }) => {
  return (
    <main className="flex-1 overflow-hidden bg-transparent relative">
      {/* Full-width and full-height container */}
      <div className="w-full h-full">
        {children}
      </div>
    </main>
  );
};

export default MainContent;