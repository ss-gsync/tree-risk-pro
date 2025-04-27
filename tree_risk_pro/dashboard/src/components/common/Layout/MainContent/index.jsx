// src/components/common/Layout/MainContent/index.jsx

import React from 'react';

const MainContent = ({ children }) => {
  return (
    <main className="flex-1 overflow-auto bg-transparent relative">
      {/* Absolute positioning ensures the map view always fills the available space */}
      <div className="absolute inset-0 w-full h-full">
        {children}
      </div>
    </main>
  );
};

export default MainContent;