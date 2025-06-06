// components/assessment/TreeAnalysis/TreeAnalysis.jsx

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Trees, AlertTriangle, Ruler, Wind, X } from 'lucide-react';

const TreeAnalysis = ({ selectedTree, onClose }) => {
  const [selectedImage, setSelectedImage] = useState(0);
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg w-3/4 h-3/4 overflow-hidden">
        <div className="flex justify-between items-center p-4 border-b">
          <h2 className="text-xl font-semibold flex items-center">
            <Trees className="h-5 w-5 mr-2 text-green-600" />
            Tree Analysis
          </h2>
          <button 
            onClick={onClose} 
            className="p-2 rounded-full hover:bg-gray-100"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        
        <div className="h-full flex">
          {/* Left Panel - Tree Images */}
          <div className="w-2/3 p-6 border-r border-gray-200">
            <div className="h-4/5 bg-gray-100 rounded-lg mb-4">
              {/* Main image viewer */}
              <div className="w-full h-full flex items-center justify-center">
                <img 
                  src="/api/placeholder/800/600" 
                  alt="Tree View" 
                  className="object-contain"
                />
              </div>
            </div>
            
            {/* Thumbnail strip */}
            <div className="flex space-x-2 overflow-x-auto">
              {[0, 1, 2, 3].map((index) => (
                <button
                  key={index}
                  className={`flex-shrink-0 w-24 h-24 rounded-lg overflow-hidden border-2 
                    ${selectedImage === index ? 'border-blue-500' : 'border-transparent'}`}
                  onClick={() => setSelectedImage(index)}
                >
                  <img 
                    src="/api/placeholder/100/100" 
                    alt={`View ${index + 1}`}
                    className="w-full h-full object-cover"
                  />
                </button>
              ))}
            </div>
          </div>

          {/* Right Panel - Analysis */}
          <div className="w-1/3 p-6 overflow-y-auto">
            <div className="space-y-6">
              {/* Tree Status */}
              <Card className="border-red-200 bg-red-50">
                <CardHeader>
                  <CardTitle className="flex items-center text-red-700">
                    <AlertTriangle className="mr-2 h-5 w-5" />
                    High Risk Assessment
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2 text-red-700">
                    <li className="flex items-center">
                      • Broken branch detected
                    </li>
                    <li className="flex items-center">
                      • Overhanging structure
                    </li>
                  </ul>
                </CardContent>
              </Card>

              {/* Measurements */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Ruler className="mr-2 h-5 w-5" />
                    Measurements
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-500">Height</span>
                      <span className="font-semibold">45 ft</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Canopy Spread</span>
                      <span className="font-semibold">30 ft</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Distance to Structure</span>
                      <span className="font-semibold">8 ft</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Historical Changes */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Wind className="mr-2 h-5 w-5" />
                    Growth Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="text-sm text-gray-500">
                      Growth rate over past year
                    </div>
                    <div className="h-24 bg-gray-100 rounded">
                      {/* Placeholder for growth chart */}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Action Buttons */}
              <div className="flex space-x-4">
                <button className="flex-1 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700">
                  Flag for Action
                </button>
                <button className="flex-1 px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700">
                  Add to Report
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TreeAnalysis;