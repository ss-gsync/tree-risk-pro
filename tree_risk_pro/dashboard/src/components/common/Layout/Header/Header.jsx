// src/components/common/Layout/Header/Header.jsx

import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Bell, User, LogOut, Settings, Menu, ChevronUp, ChevronDown, HelpCircle } from 'lucide-react';
import logo from '../../../../logo.png';
import { useAuth } from '../../../auth/AuthContext';

const Header = () => {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(true);
  const { logout } = useAuth();
  
  const toggleDropdown = () => {
    setDropdownOpen(!dropdownOpen);
  };
  
  const toggleCollapse = () => {
    setCollapsed(!collapsed);
    // Dispatch an event so other components can react to header collapse
    window.dispatchEvent(new CustomEvent('headerCollapse', { detail: { collapsed: !collapsed } }));
  };

  return (
    <header className={`bg-gradient-to-b from-red-600 to-red-800 shadow-md ${collapsed ? 'h-10' : 'h-16'} flex items-center ${collapsed ? 'justify-start' : 'justify-between'} px-6 sticky top-0 z-10 w-full left-0 transition-all duration-300`}>
      <div className="flex items-center">
        <button className="lg:hidden mr-2">
          <Menu className="h-6 w-6 text-white" />
        </button>
        
        {/* When collapsed, show admin button on the left (without text) */}
        {collapsed && (
          <div className="relative">
            <button 
              onClick={toggleDropdown}
              className="flex items-center p-1 rounded-full hover:bg-emerald-700/30"
            >
              <div className="h-7 w-7 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center border border-white/30">
                <User className="h-4 w-4 text-white" />
              </div>
            </button>
          </div>
        )}
        
        {/* When expanded, show logo and title */}
        {!collapsed && (
          <>
            <a href="https://tttdallastx.com/" target="_blank" rel="noopener noreferrer" className="flex items-center">
              <img src={logo} alt="Texas Tree Transformations" className="h-10 mr-3 hover:opacity-90 transition-opacity" />
            </a>
            <h1 className="text-2xl font-bold tracking-wider text-white drop-shadow-[0_1px_1px_rgba(0,0,0,0.4)] border-l border-white/30 pl-3">
              Tree Risk Pro<sup className="text-xs ml-1 opacity-80">©</sup>
            </h1>
          </>
        )}
      </div>
      
      <div className="flex items-center space-x-5">
        <button 
          className={`${collapsed ? 'p-1' : 'p-2'} rounded-full hover:bg-emerald-700/30`}
          onClick={toggleCollapse}
          title={collapsed ? "Expand header" : "Collapse header"}
        >
          {collapsed ? 
            <ChevronDown className="h-5 w-5 text-white" /> : 
            <ChevronUp className="h-5 w-5 text-white" />
          }
        </button>
        
        <button 
          className={`${collapsed ? 'p-1' : 'p-2'} rounded-full hover:bg-emerald-700/30`}
          onClick={() => alert("Alerts feature coming soon.")}
          title="Alerts - Coming soon"
        >
          <Bell className="h-5 w-5 text-white" />
        </button>
        
        <button 
          className={`${collapsed ? 'p-1' : 'p-2'} rounded-full hover:bg-emerald-700/30`}
          onClick={() => alert("Help documentation coming soon.")}
          title="Help"
        >
          <HelpCircle className="h-5 w-5 text-white" />
        </button>
        
        <button 
          className={`${collapsed ? 'p-1' : 'p-2'} rounded-full hover:bg-emerald-700/30`}
          onClick={() => {
            // First trigger sidebar collapse if it's not already collapsed
            window.dispatchEvent(new CustomEvent('forceSidebarCollapse', { detail: { source: 'settings' } }));
            
            // Then navigate to settings
            window.dispatchEvent(new CustomEvent('navigateTo', { detail: { view: 'Settings' } }));
          }}
          title="Settings"
        >
          <Settings className="h-5 w-5 text-white" />
        </button>
        
        {/* Admin button only shown on the right when NOT collapsed */}
        {!collapsed && (
          <div className="relative">
            <button 
              onClick={toggleDropdown}
              className="flex items-center space-x-2 p-2 rounded-full hover:bg-emerald-700/30"
            >
              <div className="h-8 w-8 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center border border-white/30">
                <User className="h-5 w-5 text-white" />
              </div>
              <span className="hidden md:block text-sm font-medium text-white">Admin</span>
            </button>
            
            {/* Dropdown menu - aligned with the admin button */}
            {dropdownOpen && (
              <div className={`absolute ${collapsed ? 'left-0' : 'right-0'} mt-2 w-64 bg-white rounded-md shadow-lg py-1 z-20`}>
                <button 
                  onClick={() => {
                    alert("Profile functionality coming soon.");
                    setDropdownOpen(false);
                  }}
                  className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 w-full text-left"
                >
                  <User className="h-4 w-4 mr-2" />
                  Profile
                  <span className="ml-auto text-xs text-emerald-600 font-medium bg-emerald-100 px-1.5 py-0.5 rounded">Beta</span>
                </button>
                <button 
                  onClick={() => {
                    alert("Settings functionality coming soon.");
                    setDropdownOpen(false);
                  }}
                  className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 w-full text-left"
                >
                  <Settings className="h-4 w-4 mr-2" />
                  Settings
                  <span className="ml-auto text-xs text-emerald-600 font-medium bg-emerald-100 px-1.5 py-0.5 rounded">Beta</span>
                </button>
                <hr className="my-1" />
                <button 
                  onClick={() => {
                    logout();
                    setDropdownOpen(false);
                  }}
                  className="flex items-center px-4 py-2 text-sm text-red-600 hover:bg-gray-100 w-full text-left"
                >
                  <LogOut className="h-4 w-4 mr-2" />
                  Logout
                </button>
              </div>
            )}
          </div>
        )}
        
        {/* Dropdown for collapsed state admin button */}
        {collapsed && dropdownOpen && (
          <div className="absolute left-0 top-10 mt-2 w-64 bg-white rounded-md shadow-lg py-1 z-20">
            <button 
              onClick={() => {
                alert("Profile functionality coming soon.");
                setDropdownOpen(false);
              }}
              className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 w-full text-left"
            >
              <User className="h-4 w-4 mr-2" />
              Profile
              <span className="ml-auto text-xs text-emerald-600 font-medium bg-emerald-100 px-1.5 py-0.5 rounded">Beta</span>
            </button>
            <button 
              onClick={() => {
                alert("Settings functionality coming soon.");
                setDropdownOpen(false);
              }}
              className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 w-full text-left"
            >
              <Settings className="h-4 w-4 mr-2" />
              Settings
              <span className="ml-auto text-xs text-emerald-600 font-medium bg-emerald-100 px-1.5 py-0.5 rounded">Beta</span>
            </button>
            <hr className="my-1" />
            <button 
              onClick={() => {
                logout();
                setDropdownOpen(false);
              }}
              className="flex items-center px-4 py-2 text-sm text-red-600 hover:bg-gray-100 w-full text-left"
            >
              <LogOut className="h-4 w-4 mr-2" />
              Logout
            </button>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;