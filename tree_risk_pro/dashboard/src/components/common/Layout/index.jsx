// src/components/common/Layout/index.jsx

import React from 'react';
import Header from './Header';
import Sidebar from './Sidebar';
import MainContent from './MainContent';

const Layout = ({ children, onNavigate, mapRef, mapDataRef }) => {
  // Handle navigation from sidebar items
  const handleNavigation = (view) => {
    if (onNavigate) {
      onNavigate(view); // Pass the view name as is, preserving case
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar 
          onNavigate={handleNavigation} 
          mapRef={mapRef} 
          mapDataRef={mapDataRef} 
        />
        <MainContent>{children}</MainContent>
      </div>
    </div>
  );
};

export default Layout;