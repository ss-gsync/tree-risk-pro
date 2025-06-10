// MLStatusIndicator.jsx
//
// Component to display the status of the ML model server

import React, { useState, useEffect } from 'react';
import { Zap, Server, AlertTriangle, Check } from 'lucide-react';

/**
 * MLStatusIndicator Component
 * 
 * Displays the status of the ML model server in the dashboard
 */
const MLStatusIndicator = () => {
  const [status, setStatus] = useState({
    loading: true,
    using_external_server: false,
    external_server_connected: false,
    cuda_available: false,
    device: '',
    model_type: '',
    external_server_url: ''
  });

  // Fetch ML engine status when component mounts
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch('/api/ml/status');
        if (response.ok) {
          const responseData = await response.json();
          // Extract the actual status object from the nested response
          const data = responseData.status || responseData;
          
          setStatus({
            loading: false,
            ...data
          });
          
          // Store the status in a global variable for other components to access
          window.mlEngineStatus = data;
          
          // Dispatch event for other components that might need the status
          window.dispatchEvent(new CustomEvent('mlEngineStatusUpdated', {
            detail: data
          }));
          
          console.log('MLStatusIndicator: ML engine status updated', data);
        } else {
          setStatus({
            loading: false,
            error: 'Failed to fetch status'
          });
        }
      } catch (error) {
        setStatus({
          loading: false,
          error: error.message
        });
      }
    };

    fetchStatus();
    
    // Refresh status every 30 seconds
    const intervalId = setInterval(fetchStatus, 30000);
    
    // Also refresh when T4 connection status might have changed
    const handleStatusChangeRequest = () => {
      console.log('MLStatusIndicator: Refreshing status due to external request');
      fetchStatus();
    };
    
    window.addEventListener('refreshMLEngineStatus', handleStatusChangeRequest);
    
    return () => {
      clearInterval(intervalId);
      window.removeEventListener('refreshMLEngineStatus', handleStatusChangeRequest);
    };
  }, []);
  
  // If loading, show loading indicator
  if (status.loading) {
    return (
      <div className="inline-flex items-center gap-1 text-gray-500 text-xs font-mono px-2 py-1 rounded">
        <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-gray-500"></div>
        <span>Loading...</span>
      </div>
    );
  }
  
  // If error, show error message
  if (status.error) {
    return (
      <div className="inline-flex items-center gap-1 text-amber-600 text-xs font-mono px-2 py-1 bg-amber-50 rounded">
        <AlertTriangle size={12} />
        <span>ML Status Error</span>
      </div>
    );
  }
  
  // Determine display based on status
  let backgroundColor = 'bg-gray-100';
  let textColor = 'text-gray-700';
  let icon = null;
  let label = 'CPU Mode';
  
  if (status.using_external_server) {
    if (status.external_server_connected) {
      backgroundColor = 'bg-green-50';
      textColor = 'text-green-700';
      icon = <Server size={12} className="text-green-600" />;
      label = 'GPU Active';
    } else {
      backgroundColor = 'bg-amber-50';
      textColor = 'text-amber-700';
      icon = <AlertTriangle size={12} className="text-amber-600" />;
      label = 'Server Disconnected';
    }
  } else if (status.cuda_available) {
    backgroundColor = 'bg-blue-50';
    textColor = 'text-blue-700';
    icon = <Zap size={12} className="text-blue-600" />;
    label = 'Local GPU';
  }
  
  // Render status indicator
  return (
    <div className="group relative inline-block">
      <div className={`inline-flex items-center gap-1 ${textColor} text-xs font-mono px-2 py-1 ${backgroundColor} rounded border border-${backgroundColor.replace('bg-', 'border-')} shadow-sm cursor-pointer`}>
        {icon}
        <span>{label}</span>
      </div>
      
      {/* Show details on hover */}
      <div className="absolute bottom-full right-0 hidden group-hover:block z-50 min-w-64 p-3 bg-white shadow-lg rounded-md border border-gray-200 mt-1 transform -translate-y-1">
        <div className="text-xs space-y-1.5 text-gray-700">
          <div className="font-semibold border-b pb-1 mb-1">ML Engine Details</div>
          <div className="flex justify-between">
            <span>Engine:</span>
            <span className="font-mono">{status.model_type || 'Unknown'}</span>
          </div>
          <div className="flex justify-between">
            <span>Device:</span>
            <span className="font-mono">{status.device || 'Unknown'}</span>
          </div>
          {status.using_external_server && (
            <>
              <div className="flex justify-between">
                <span>T4 Server:</span>
                <span className="font-mono truncate max-w-32">
                  {status.external_server_url?.replace(/^https?:\/\//, '') || 'Unknown'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Connection:</span>
                <span className={`flex items-center ${status.external_server_connected ? 'text-green-600' : 'text-amber-600'}`}>
                  {status.external_server_connected ? (
                    <><Check size={10} className="mr-1" /> Connected</>
                  ) : (
                    <><AlertTriangle size={10} className="mr-1" /> Disconnected</>
                  )}
                </span>
              </div>
            </>
          )}
          <div className="flex justify-between">
            <span>Status:</span>
            <span className="flex items-center">
              {status.external_server_connected || status.cuda_available ? (
                <><Check size={10} className="text-green-600 mr-1" /> Ready</>
              ) : (
                <><AlertTriangle size={10} className="text-amber-600 mr-1" /> Limited</>
              )}
            </span>
          </div>
          
          {/* Performance information */}
          <div className="mt-2 pt-1 border-t border-gray-100">
            <div className="font-medium mb-1 text-gray-600">Performance Impact</div>
            <p className="text-gray-600 text-xs leading-tight">
              {status.external_server_connected ? (
                "T4 GPU acceleration provides up to 10x faster tree detection."
              ) : status.cuda_available ? (
                "Local GPU acceleration is active for improved performance."
              ) : (
                "Running on CPU only. GPU acceleration recommended for best performance."
              )}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MLStatusIndicator;