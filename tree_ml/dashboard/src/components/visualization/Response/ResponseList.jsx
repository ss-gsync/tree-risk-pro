// src/components/visualization/Response/ResponseList.jsx
//
// Component for displaying and managing a list of detection responses
// Allows browsing through detection results and selecting one for viewing

import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../../../ui/card';
import { ScrollArea } from '../../../ui/scroll-area';
import { Button } from '../../../ui/button';
import { Input } from '../../../ui/input';
import { 
  Search, List, Calendar, Inbox, AlertCircle, FileText,
  ChevronLeft, ChevronRight, RefreshCw, Map, Filter
} from 'lucide-react';

/**
 * ResponseList Component
 * 
 * Displays a list of available detection responses and allows selecting one for viewing
 */
const ResponseList = ({ 
  onSelectResponse,
  selectedDetectionId,
  refreshInterval = 30000 // 30 seconds default refresh interval
}) => {
  const [responses, setResponses] = useState([]);
  const [filteredResponses, setFilteredResponses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortOrder, setSortOrder] = useState('newest');
  const [page, setPage] = useState(0);
  const [pageSize] = useState(10);

  // Load detection responses from the server
  const loadResponses = async () => {
    try {
      setLoading(true);
      
      // Use directory listing to find all detection_* directories
      const response = await fetch('/api/detections');
      
      if (!response.ok) {
        throw new Error(`Failed to load detections: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Process response data
      const detections = data.detections || [];
      
      // Add additional metadata if available
      const processedDetections = detections.map(detection => {
        // Extract timestamp from job ID if not provided
        if (!detection.timestamp && detection.job_id) {
          const timestampMatch = detection.job_id.match(/detection_(\d+)/);
          if (timestampMatch && timestampMatch[1]) {
            const timestamp = parseInt(timestampMatch[1]);
            if (!isNaN(timestamp)) {
              detection.timestamp = new Date(timestamp).toISOString();
            }
          }
        }
        
        return detection;
      });
      
      // Sort by timestamp (newest first by default)
      const sortedDetections = processedDetections.sort((a, b) => {
        const dateA = a.timestamp ? new Date(a.timestamp) : new Date(0);
        const dateB = b.timestamp ? new Date(b.timestamp) : new Date(0);
        return dateB - dateA;
      });
      
      setResponses(sortedDetections);
      filterAndSortResponses(sortedDetections, searchTerm, sortOrder);
      setLoading(false);
    } catch (err) {
      console.error("Error loading detection responses:", err);
      setError(err.message);
      setLoading(false);
    }
  };

  // Filter and sort responses based on search term and sort order
  const filterAndSortResponses = (allResponses, search, order) => {
    // Filter by search term
    let filtered = allResponses;
    if (search) {
      const lowerSearch = search.toLowerCase();
      filtered = allResponses.filter(response => 
        (response.job_id && response.job_id.toLowerCase().includes(lowerSearch)) ||
        (response.timestamp && response.timestamp.toLowerCase().includes(lowerSearch)) ||
        (response.detection_count !== undefined && response.detection_count.toString().includes(lowerSearch))
      );
    }
    
    // Sort by timestamp
    filtered = [...filtered].sort((a, b) => {
      const dateA = a.timestamp ? new Date(a.timestamp) : new Date(0);
      const dateB = b.timestamp ? new Date(b.timestamp) : new Date(0);
      return order === 'newest' ? dateB - dateA : dateA - dateB;
    });
    
    setFilteredResponses(filtered);
  };

  // Load responses on component mount and start refresh interval
  useEffect(() => {
    loadResponses();
    
    // Set up automatic refresh
    const intervalId = setInterval(loadResponses, refreshInterval);
    
    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, [refreshInterval]);

  // Handle search term change
  useEffect(() => {
    filterAndSortResponses(responses, searchTerm, sortOrder);
  }, [searchTerm, sortOrder, responses]);

  // Handle sort order change
  const handleSortOrderChange = () => {
    const newOrder = sortOrder === 'newest' ? 'oldest' : 'newest';
    setSortOrder(newOrder);
  };

  // Format date for display
  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown';
    
    try {
      const date = new Date(dateString);
      return date.toLocaleString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch (err) {
      return dateString;
    }
  };

  // Calculate pagination
  const totalPages = Math.ceil(filteredResponses.length / pageSize);
  const paginatedResponses = filteredResponses.slice(
    page * pageSize, 
    (page + 1) * pageSize
  );

  // Next/Previous page handlers
  const handleNextPage = () => {
    setPage(prevPage => Math.min(prevPage + 1, totalPages - 1));
  };

  const handlePrevPage = () => {
    setPage(prevPage => Math.max(prevPage - 1, 0));
  };

  // Render error state
  if (error) {
    return (
      <Card className="w-full shadow-md">
        <CardHeader className="pb-2">
          <CardTitle className="text-red-500 flex items-center text-sm">
            <AlertCircle className="h-4 w-4 mr-2" />
            Error Loading Detections
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-red-500">{error}</p>
          <Button 
            variant="outline" 
            size="sm" 
            className="mt-4"
            onClick={loadResponses}
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  // Render loading state
  if (loading && responses.length === 0) {
    return (
      <Card className="w-full shadow-md">
        <CardHeader className="pb-2">
          <CardTitle className="text-md flex items-center">
            <List className="h-4 w-4 mr-2" />
            Loading Detections...
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Render empty state
  if (responses.length === 0) {
    return (
      <Card className="w-full shadow-md">
        <CardHeader className="pb-2">
          <CardTitle className="text-md flex items-center">
            <Inbox className="h-4 w-4 mr-2" />
            No Detections Found
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm">No detection results are available.</p>
          <Button 
            variant="outline" 
            size="sm" 
            className="mt-4"
            onClick={loadResponses}
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </CardContent>
      </Card>
    );
  }

  // Render list of responses
  return (
    <Card className="w-full shadow-md">
      <CardHeader className="pb-2">
        <CardTitle className="text-md flex items-center justify-between">
          <div className="flex items-center">
            <List className="h-4 w-4 mr-2" />
            Detection Results
            {loading && (
              <RefreshCw className="h-3 w-3 ml-2 animate-spin opacity-70" />
            )}
          </div>
          <Button 
            variant="ghost"
            size="sm"
            onClick={loadResponses}
            title="Refresh"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        </CardTitle>
      </CardHeader>
      
      <CardContent>
        {/* Search and filter */}
        <div className="flex items-center mb-3 gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-gray-400" />
            <Input
              placeholder="Search detections..."
              className="pl-8"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={handleSortOrderChange}
            title={sortOrder === 'newest' ? "Newest first" : "Oldest first"}
          >
            <Calendar className="h-4 w-4 mr-1" />
            {sortOrder === 'newest' ? "↓" : "↑"}
          </Button>
        </div>
        
        {/* No results message */}
        {filteredResponses.length === 0 && (
          <div className="py-6 text-center">
            <Inbox className="h-10 w-10 mx-auto text-gray-400 mb-2" />
            <p className="text-sm text-gray-500">No matching detection results</p>
            {searchTerm && (
              <Button 
                variant="ghost"
                size="sm"
                className="mt-2"
                onClick={() => setSearchTerm('')}
              >
                Clear search
              </Button>
            )}
          </div>
        )}
        
        {/* Response list */}
        {filteredResponses.length > 0 && (
          <ScrollArea className="h-[300px] pr-2">
            <div className="space-y-2">
              {paginatedResponses.map((response) => (
                <div
                  key={response.job_id}
                  className={`p-2 rounded-md cursor-pointer border transition-colors ${
                    selectedDetectionId === response.job_id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:bg-gray-50'
                  }`}
                  onClick={() => onSelectResponse(response.job_id)}
                >
                  <div className="flex justify-between items-start">
                    <div className="flex items-center">
                      <FileText className="h-4 w-4 mr-2 text-gray-500" />
                      <div className="truncate max-w-[180px]">
                        {response.job_id}
                      </div>
                    </div>
                    {response.detection_count !== undefined && (
                      <div className="text-xs bg-gray-100 px-2 py-0.5 rounded-full">
                        {response.detection_count} trees
                      </div>
                    )}
                  </div>
                  <div className="flex justify-between items-center mt-1">
                    <div className="text-xs text-gray-500">
                      {formatDate(response.timestamp)}
                    </div>
                    {response.source && (
                      <div className="text-xs text-gray-500">
                        {response.source}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        )}
        
        {/* Pagination */}
        {filteredResponses.length > pageSize && (
          <div className="flex items-center justify-between mt-4">
            <div className="text-xs text-gray-500">
              Showing {page * pageSize + 1}-{Math.min((page + 1) * pageSize, filteredResponses.length)} of {filteredResponses.length}
            </div>
            <div className="flex gap-1">
              <Button
                variant="outline"
                size="sm"
                onClick={handlePrevPage}
                disabled={page === 0}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleNextPage}
                disabled={page >= totalPages - 1}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ResponseList;