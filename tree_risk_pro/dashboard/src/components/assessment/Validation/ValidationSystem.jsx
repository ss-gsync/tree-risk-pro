// components/assessment/Validation/ValidationSystem.jsx
// This component provides a full validation workflow for tree assessments 
// and works with the MapView's validation mode

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../../components/ui/card';
import { 
  CheckCircle, AlertTriangle, Camera, 
  FileText, Clock, ArrowRight, Download,
  Printer, Mail, ChevronRight, Activity,
  Calendar, MapPin, Ruler, X, Home, Info
} from 'lucide-react';
import { PropertyService, ValidationService } from '../../../services/api/apiService';

const ValidationSystem = ({ selectedTree, onClose }) => {
  const [validationStep, setValidationStep] = useState(0);
  const [assessmentNotes, setAssessmentNotes] = useState({});
  const [showReport, setShowReport] = useState(false);
  const [riskLevel, setRiskLevel] = useState('no_risk'); // Default to no_risk
  const [propertyInfo, setPropertyInfo] = useState(null);
  const [lidarData, setLidarData] = useState(null);
  const [loadingLidar, setLoadingLidar] = useState(false);
  
  // Fetch property information when selectedTree changes
  useEffect(() => {
    const fetchPropertyData = async () => {
      if (selectedTree?.property_id) {
        try {
          const property = await PropertyService.getProperty(selectedTree.property_id);
          setPropertyInfo(property);
        } catch (error) {
          console.error('Error fetching property:', error);
        }
      }
    };
    
    fetchPropertyData();
  }, [selectedTree]);
  
  // Fetch LiDAR data when selectedTree changes
  useEffect(() => {
    const fetchLidarData = async () => {
      if (selectedTree?.tree_id) {
        setLoadingLidar(true);
        try {
          const data = await ValidationService.getTreeLidarData(selectedTree.tree_id);
          setLidarData(data);
          
          // Also set risk level based on the tree's assessed risk
          if (selectedTree.risk_level && selectedTree.risk_level !== 'new') {
            setRiskLevel(selectedTree.risk_level);
          } else {
            // Default to no_risk when no risk level is specified or when it's 'new'
            setRiskLevel('no_risk');
          }
        } catch (error) {
          console.error('Error fetching LiDAR data:', error);
          setLidarData(null);
        } finally {
          setLoadingLidar(false);
        }
      }
    };
    
    fetchLidarData();
  }, [selectedTree]);
  
  // Notify map view when validation system is mounted/unmounted
  useEffect(() => {
    // When component mounts, dispatch an event to notify map view
    const validationActiveEvent = new CustomEvent('validationSystemActive', {
      detail: { 
        active: true,
        treeId: selectedTree?.tree_id
      }
    });
    window.dispatchEvent(validationActiveEvent);
    
    // When component unmounts, notify map view that validation is inactive
    return () => {
      const validationInactiveEvent = new CustomEvent('validationSystemActive', {
        detail: { 
          active: false 
        }
      });
      window.dispatchEvent(validationInactiveEvent);
    };
  }, [selectedTree?.tree_id]);
  
  const validationSteps = [
    {
      id: 'visual',
      title: 'Visual Confirmation',
      checks: [
        'Broken branch location matches LiDAR',
        'Structure proximity verified',
        'Tree species identification correct'
      ]
    },
    {
      id: 'measurements',
      title: 'Measurement Validation',
      checks: [
        'Height measurement within 10%',
        'Branch thickness accurate',
        'Distance to structure verified'
      ]
    },
    {
      id: 'risk',
      title: 'Risk Assessment',
      checks: [
        'Failure potential evaluated',
        'Target exposure confirmed',
        'Risk level appropriate'
      ]
    }
  ];

  const handleCheckToggle = (stepId, check) => {
    setAssessmentNotes(prev => {
      const stepChecks = prev[stepId] || [];
      if (stepChecks.includes(check)) {
        return {
          ...prev,
          [stepId]: stepChecks.filter(item => item !== check)
        };
      } else {
        return {
          ...prev,
          [stepId]: [...stepChecks, check]
        };
      }
    });
  };

  const handleNextStep = () => {
    if (validationStep < validationSteps.length - 1) {
      setValidationStep(validationStep + 1);
    } else {
      setShowReport(true);
    }
  };

  const handlePrevStep = () => {
    if (validationStep > 0) {
      setValidationStep(validationStep - 1);
    }
  };
  
  const handleCompleteValidation = async () => {
    if (!selectedTree) return;
    
    try {
      // Mark the tree as validated in the backend
      await ValidationService.updateValidationStatus(selectedTree.id, 'approved');
      
      // Convert the assessment notes into a structured format
      const completedChecks = Object.entries(assessmentNotes).reduce((acc, [stepId, checks]) => {
        return {
          ...acc,
          [stepId]: {
            completed: true,
            checks: checks
          }
        };
      }, {});
      
      // Update the tree with completed validation data
      const validationData = {
        tree_id: selectedTree.tree_id,
        validation_date: new Date().toISOString(),
        risk_level: riskLevel || 'no_risk', // Ensure we always have a risk_level, default to no_risk
        validator_notes: assessmentNotes,
        completed_checks: completedChecks
      };
      
      // Save the validation data
      await ValidationService.saveValidationData(selectedTree.id, validationData);
      
      // Notify map view that validation is complete
      window.dispatchEvent(new CustomEvent('treeValidationComplete', {
        detail: {
          treeId: selectedTree.tree_id,
          status: 'approved',
          riskLevel: riskLevel,
          validationData: validationData
        }
      }));
      
      // Show the validation report
      setShowReport(true);
      
      // Exit validation mode after a delay
      setTimeout(() => {
        onClose();
      }, 5000);
    } catch (error) {
      console.error('Error completing validation:', error);
    }
  };

  const isStepComplete = (stepId) => {
    const stepChecks = assessmentNotes[stepId] || [];
    const totalChecks = validationSteps.find(s => s.id === stepId).checks.length;
    return stepChecks.length === totalChecks;
  };

  // Render main validation interface
  const ValidationInterface = () => (
    <div className="flex flex-1">
      {/* Left panel - Tree view and info */}
      <div className="w-1/2 p-6 border-r border-gray-200">
        <div className="mb-6">
          <h3 className="text-lg font-medium mb-2">
            {selectedTree?.tree_species} - {selectedTree?.tree_height}ft
          </h3>
          <div className="flex items-center text-gray-600 mb-2">
            <MapPin className="w-4 h-4 mr-1" />
            <span className="text-sm">
              {(() => {
                // Extract location from description
                const parts = selectedTree?.location?.description?.split(' - ');
                let locationDesc = '';
                
                if (parts && parts.length > 1) {
                  // If there are multiple parts, the last part is typically the location
                  locationDesc = parts[parts.length - 1];
                } else if (selectedTree?.location?.description && !selectedTree?.location?.description.includes(selectedTree?.tree_species)) {
                  // If description doesn't include species, use whole description
                  locationDesc = selectedTree?.location?.description;
                }
                
                // Format for display
                if (!locationDesc) {
                  return propertyInfo?.address || 'Unknown address';
                }
                
                // Remove "of property" if it exists to prevent "of property of address"
                if (locationDesc.toLowerCase().endsWith('of property')) {
                  locationDesc = locationDesc.substring(0, locationDesc.length - 11);
                }
                
                // Check if locationDesc already contains the property address to avoid duplication
                if (propertyInfo?.address && locationDesc.includes(propertyInfo.address)) {
                  return locationDesc;
                } else {
                  return propertyInfo?.address || locationDesc || 'Unknown address';
                }
              })()}
            </span>
          </div>
          <div className="flex space-x-2">
            <span className="px-2 py-1 bg-amber-100 text-amber-800 rounded-full text-sm flex items-center">
              <Clock className="w-4 h-4 mr-1" />
              Pending
            </span>
            {selectedTree?.riskFactors.some(f => f.level === 'high') && (
              <span className="px-2 py-1 bg-red-100 text-red-800 rounded-full text-sm flex items-center">
                <AlertTriangle className="w-4 h-4 mr-1" />
                High Risk
              </span>
            )}
          </div>
        </div>

        {/* Tree details */}
        <Card className="mb-4 border rounded shadow-sm">
          <CardHeader className="py-3 border-b bg-gray-50">
            <CardTitle className="text-sm">Tree Details</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-gray-500">Species</span>
              <span className="font-medium">{selectedTree?.tree_species || 'Unknown'}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-500">Height</span>
              <span className="font-medium">{selectedTree?.tree_height || 'Unknown'} ft</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-500">Canopy Width</span>
              <span className="font-medium">{selectedTree?.canopy_width || 'Unknown'} ft</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-500">DBH</span>
              <span className="font-medium">{selectedTree?.dbh || selectedTree?.tree_diameter || 'Unknown'} in</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-500">Condition</span>
              <span className="font-medium">{selectedTree?.tree_condition || 'Unknown'}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-500">Proximity</span>
              <span className="font-medium">{selectedTree?.proximity || 'Unknown'}</span>
            </div>
          </CardContent>
        </Card>

        {/* LiDAR Data */}
        <div className="mb-4">
          <h4 className="text-sm font-medium mb-2 flex items-center justify-between">
            <div className="flex items-center">
              <Activity className="w-4 h-4 mr-1" />
              LiDAR Data {loadingLidar && '(Loading...)'}
            </div>
            {lidarData && (
              <span className="text-xs bg-blue-50 text-blue-600 px-2 py-0.5 rounded-full">
                Scan date: {lidarData.scan_date}
              </span>
            )}
          </h4>
          
          {lidarData ? (
            <div className="rounded border overflow-hidden shadow-sm">
              <div className="p-3 bg-gray-50 border-b">
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Point Count:</span>
                    <span>{lidarData.point_count?.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Scan Type:</span>
                    <span>{lidarData.scan_type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Trunk Volume:</span>
                    <span>{lidarData.trunk_volume?.toFixed(1)} ft³</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Canopy Volume:</span>
                    <span>{lidarData.canopy_volume?.toFixed(1)} ft³</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Canopy Area:</span>
                    <span>{lidarData.canopy_area?.toFixed(1)} ft²</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Carbon Seq.:</span>
                    <span>{lidarData.carbon_sequestration?.toFixed(1)} kg</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-40 bg-gray-100 rounded-lg flex items-center justify-center">
              {loadingLidar ? (
                <div className="text-gray-400 flex flex-col items-center">
                  <div className="animate-spin h-8 w-8 border-2 border-gray-500 border-t-transparent rounded-full mb-2"></div>
                  <span>Loading LiDAR data...</span>
                </div>
              ) : (
                <div className="text-gray-400 flex flex-col items-center">
                  <Info className="h-8 w-8 mb-2" />
                  <span>No LiDAR data available</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* LiDAR view */}
        <div className="h-64 bg-gray-100 rounded border border-gray-300 mb-4 flex items-center justify-center flex-col shadow-sm">
          <img src="/api/placeholder/400/200" alt="LiDAR view" className="object-contain" />
          <span className="text-xs text-gray-500 mt-2">LiDAR view coming soon</span>
        </div>
      </div>

      {/* Right panel - Validation steps */}
      <div className="w-1/2 p-6">
        <div className="mb-6">
          <div className="flex items-center mb-4">
            {validationSteps.map((step, index) => (
              <React.Fragment key={step.id}>
                <div 
                  className={`flex items-center justify-center w-8 h-8 rounded-full ${
                    index < validationStep 
                      ? 'bg-green-500 text-white' 
                      : index === validationStep 
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-200 text-gray-600'
                  }`}
                >
                  {index < validationStep ? (
                    <CheckCircle className="w-5 h-5" />
                  ) : (
                    <span>{index + 1}</span>
                  )}
                </div>
                {index < validationSteps.length - 1 && (
                  <div 
                    className={`h-1 flex-1 ${
                      index < validationStep ? 'bg-green-500' : 'bg-gray-200'
                    }`}
                  />
                )}
              </React.Fragment>
            ))}
          </div>
          
          <h3 className="text-xl font-medium mb-1">
            {validationSteps[validationStep].title}
          </h3>
          <p className="text-gray-500 text-sm">
            {validationStep === 0 && "Compare visual evidence with LiDAR data"}
            {validationStep === 1 && "Verify that measurements and species are accurate"}
            {validationStep === 2 && "Confirm the overall risk assessment"}
          </p>
        </div>

        {/* Validation checks for current step */}
        <Card className="mb-6 border rounded shadow-sm">
          <CardContent className="p-4">
            {validationSteps[validationStep].checks.map((check, checkIndex) => (
              <div 
                key={checkIndex}
                className="flex items-center justify-between p-3 border-b last:border-0"
              >
                <span>{check}</span>
                <button
                  onClick={() => handleCheckToggle(validationSteps[validationStep].id, check)}
                  className={`w-6 h-6 rounded flex items-center justify-center ${
                    (assessmentNotes[validationSteps[validationStep].id] || []).includes(check)
                      ? 'bg-green-500 text-white border border-green-600'
                      : 'bg-gray-200 border border-gray-300'
                  }`}
                >
                  {(assessmentNotes[validationSteps[validationStep].id] || []).includes(check) && (
                    <CheckCircle className="w-4 h-4" />
                  )}
                </button>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Notes area */}
        <div className="mb-6">
          <h4 className="text-sm font-medium mb-2">Notes</h4>
          <textarea
            className="w-full border rounded p-3 h-24 resize-none bg-gray-50 text-gray-800 focus:bg-white focus:ring-1 focus:ring-blue-500 focus:border-blue-500 focus:outline-none"
            placeholder="Add any additional notes about this validation step..."
          />
        </div>

        {/* Risk level selection (only in risk assessment step) */}
        {validationStep === 2 && (
          <div className="mb-6">
            <h4 className="text-sm font-medium mb-2">Risk Level Assessment</h4>
            <div className="flex space-x-2">
              <button
                onClick={() => setRiskLevel('no_risk')}
                className={`flex-1 py-2 px-4 rounded border ${
                  riskLevel === 'no_risk' 
                    ? 'bg-green-200 text-green-700 border-green-300' 
                    : 'bg-green-100 text-green-700 hover:bg-green-200 border-green-200'
                }`}
              >
                No Risk
              </button>
              <button
                onClick={() => setRiskLevel('low')}
                className={`flex-1 py-2 px-4 rounded border ${
                  riskLevel === 'low' 
                    ? 'bg-yellow-200 text-yellow-700 border-yellow-300' 
                    : 'bg-yellow-100 text-yellow-700 hover:bg-yellow-200 border-yellow-200'
                }`}
              >
                Low
              </button>
              <button
                onClick={() => setRiskLevel('medium')}
                className={`flex-1 py-2 px-4 rounded border ${
                  riskLevel === 'medium' 
                    ? 'bg-orange-200 text-orange-700 border-orange-300' 
                    : 'bg-orange-100 text-orange-700 hover:bg-orange-200 border-orange-200'
                }`}
              >
                Medium
              </button>
              <button
                onClick={() => setRiskLevel('high')}
                className={`flex-1 py-2 px-4 rounded border ${
                  riskLevel === 'high' 
                    ? 'bg-red-200 text-red-700 border-red-300' 
                    : 'bg-red-100 text-red-700 hover:bg-red-200 border-red-200'
                }`}
              >
                High
              </button>
            </div>
          </div>
        )}

        {/* Navigation buttons */}
        <div className="flex justify-between">
          <button
            onClick={handlePrevStep}
            className={`px-4 py-2 rounded border ${
              validationStep === 0 ? 'bg-gray-100 text-gray-400 cursor-not-allowed border-gray-200' : 'bg-gray-100 hover:bg-gray-200 border-gray-300'
            }`}
            disabled={validationStep === 0}
          >
            Previous
          </button>
          <button
            onClick={handleNextStep}
            className="px-4 py-2 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded border border-blue-200 flex items-center"
          >
            {validationStep === validationSteps.length - 1 ? 'Complete Validation' : 'Next'}
            <ArrowRight className="w-4 h-4 ml-1" />
          </button>
        </div>
      </div>
    </div>
  );

  const ReportPreview = () => (
    <div>
      <div className="p-6 border-b border-gray-200">
        <div className="flex justify-between items-center mb-4">
          <button
            onClick={handleCompleteValidation}
            className="px-4 py-2 flex items-center bg-blue-50 hover:bg-blue-100 text-blue-700 rounded border border-blue-200"
          >
            Generate Report
          </button>
          <div className="flex space-x-4">
          {/* Print button removed */}
          <button 
            onClick={() => alert("Email functionality coming soon.")}
            className="px-4 py-2 flex items-center text-gray-600 hover:bg-gray-100 rounded border border-gray-200"
          >
            <Mail className="h-5 w-5 mr-2" />
            Email
          </button>
          <button 
            onClick={() => alert("PDF download functionality coming soon.")}
            className="px-4 py-2 flex items-center text-gray-600 hover:bg-gray-100 rounded border border-gray-200"
          >
            <Download className="h-5 w-5 mr-2" />
            Download PDF
          </button>
          <button
            onClick={() => setShowReport(false)}
            className="px-4 py-2 flex items-center text-gray-600 hover:bg-gray-100 rounded border border-gray-200"
          >
            Edit
          </button>
        </div>
        </div>
        
        {/* Risk Level Assessment buttons removed */}
      </div>

      <div className="flex-1 p-6 overflow-y-auto">
        {/* Report Content */}
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Overview Section */}
          <Card>
            <CardHeader>
              <CardTitle>Property Overview</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-6">
                <div className="flex items-center">
                  <MapPin className="h-5 w-5 mr-2 text-gray-400" />
                  <div>
                    <div className="text-sm text-gray-500">Location</div>
                    <div className="font-medium">
                      {(() => {
                        // Extract location from description or use property address
                        const parts = selectedTree?.location?.description?.split(' - ');
                        let locationDesc = '';
                        
                        if (parts && parts.length > 1) {
                          // If there are multiple parts, the last part is typically the location
                          locationDesc = parts[parts.length - 1];
                        } else if (selectedTree?.location?.description && !selectedTree?.location?.description.includes(selectedTree?.tree_species)) {
                          // If description doesn't include species, use whole description
                          locationDesc = selectedTree?.location?.description;
                        }
                        
                        // Clean up location description
                        if (locationDesc.toLowerCase().endsWith('of property')) {
                          locationDesc = locationDesc.substring(0, locationDesc.length - 11);
                        }
                        
                        // Use property address if available
                        const address = propertyInfo?.address || locationDesc || 'Unknown address';
                        
                        // Split address into street and city parts
                        const addressParts = address.split(',');
                        const street = addressParts[0] || '';
                        const city = addressParts.length > 1 ? addressParts.slice(1).join(',').trim() : '';
                        
                        return (
                          <>
                            <div>{street}</div>
                            <div className="text-gray-600">{city}</div>
                          </>
                        );
                      })()}
                    </div>
                  </div>
                </div>
                <div className="flex items-center">
                  <Calendar className="h-5 w-5 mr-2 text-gray-400" />
                  <div>
                    <div className="text-sm text-gray-500">Assessment Date</div>
                    <div className="font-medium">{lidarData?.scan_date || 'February 18, 2025'}</div>
                  </div>
                </div>
                <div className="flex items-center">
                  <Activity className="h-5 w-5 mr-2 text-gray-400" />
                  <div>
                    <div className="text-sm text-gray-500">Risk Level</div>
                    <div className={`font-medium ${
                      riskLevel === 'high'
                        ? 'text-red-600'
                        : riskLevel === 'medium'
                          ? 'text-orange-600'
                          : 'text-green-600'
                    }`}>
                      {riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1)}
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Tree Details */}
          <Card>
            <CardHeader>
              <CardTitle>Tree Specifications</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium mb-4">Measurements</h4>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500">Height</span>
                      <span className="font-medium">{selectedTree?.tree_height || 'Unknown'} feet</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500">Canopy Spread</span>
                      <span className="font-medium">{selectedTree?.canopy_width || 'Unknown'} feet</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500">DBH</span>
                      <span className="font-medium">{selectedTree?.dbh || selectedTree?.tree_diameter || 'Unknown'} inches</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500">Condition</span>
                      <span className="font-medium">{selectedTree?.tree_condition || 'Unknown'}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500">Canopy Density</span>
                      <span className="font-medium">{selectedTree?.canopy_density || 'Unknown'}</span>
                    </div>
                    {lidarData?.biomass_estimate && (
                      <div className="flex items-center justify-between">
                        <span className="text-gray-500">Biomass Estimate</span>
                        <span className="font-medium">{lidarData.biomass_estimate.toFixed(1)} kg</span>
                      </div>
                    )}
                    {lidarData?.carbon_sequestration && (
                      <div className="flex items-center justify-between">
                        <span className="text-gray-500">Carbon Sequestration</span>
                        <span className="font-medium">{lidarData.carbon_sequestration.toFixed(1)} kg</span>
                      </div>
                    )}
                  </div>
                </div>
                <div>
                  <h4 className="font-medium mb-4">Risk Factors</h4>
                  <div className="space-y-2">
                    {selectedTree?.riskFactors.map((factor, idx) => (
                      <div 
                        key={idx}
                        className={`flex items-center ${
                          factor.level === 'high'
                            ? 'text-red-600'
                            : factor.level === 'medium'
                              ? 'text-orange-600'
                              : 'text-green-600'
                        }`}
                      >
                        <AlertTriangle className="h-4 w-4 mr-2" />
                        {factor.description}
                      </div>
                    ))}
                    
                    {/* Show LiDAR-specific risk indicators if available */}
                    {lidarData?.risk_indicators && Object.entries(lidarData.risk_indicators)
                      .filter(([_, indicator]) => indicator.detected)
                      .map(([type, indicator]) => (
                        <div 
                          key={type}
                          className={`flex items-center ${
                            indicator.severity === 'high'
                              ? 'text-red-600'
                              : indicator.severity === 'medium'
                                ? 'text-orange-600'
                                : 'text-green-600'
                          }`}
                        >
                          <AlertTriangle className="h-4 w-4 mr-2" />
                          <span>LiDAR: {type.replace('_', ' ')} - {indicator.location || ''}</span>
                        </div>
                      ))
                    }
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Visual Evidence */}
          <Card>
            <CardHeader>
              <CardTitle>Visual Evidence</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="aspect-video bg-gray-100 rounded-lg flex items-center justify-center">
                    {lidarData?.thumbnail_url ? (
                      <div className="text-gray-600 flex flex-col items-center">
                        <img 
                          src={lidarData.thumbnail_url} 
                          alt="LiDAR view" 
                          className="object-contain h-full opacity-80" 
                        />
                        <span className="absolute bg-black bg-opacity-50 text-white px-2 py-1 rounded text-xs">
                          LiDAR View
                        </span>
                      </div>
                    ) : (
                      <div className="text-gray-400 flex flex-col items-center">
                        <Activity className="h-10 w-10 mb-2" />
                        <span>LiDAR data visualization</span>
                      </div>
                    )}
                  </div>
                  <p className="text-sm text-gray-500">
                    {lidarData ? 
                      `LiDAR Scan (${lidarData.point_count?.toLocaleString()} points)` : 
                      'LiDAR Scan'}
                  </p>
                </div>
                <div className="space-y-4">
                  <div className="aspect-video bg-gray-100 rounded-lg flex items-center justify-center">
                    {lidarData?.model_url ? (
                      <div className="text-gray-600 flex flex-col items-center">
                        <img 
                          src={lidarData.thumbnail_url || "/mock/frames/tree-model-placeholder.png"} 
                          alt="RGB view" 
                          className="object-contain h-full opacity-80" 
                        />
                        <span className="absolute bg-black bg-opacity-50 text-white px-2 py-1 rounded text-xs">
                          RGB View
                        </span>
                      </div>
                    ) : (
                      <div className="text-gray-400 flex flex-col items-center">
                        <Camera className="h-10 w-10 mb-2" />
                        <span>Aerial view of tree position</span>
                      </div>
                    )}
                  </div>
                  <p className="text-sm text-gray-500">
                    {selectedTree?.tree_species || 'Tree'} - {selectedTree?.tree_height || ''} ft height
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Recommendations */}
          <Card>
            <CardHeader>
              <CardTitle>Recommendations</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                  <h4 className="font-medium text-red-700 mb-2">Immediate Action Suggested</h4>
                  <ul className="space-y-2 text-red-600">
                    <li className="flex items-start">
                      <ChevronRight className="h-5 w-5 mr-2 flex-shrink-0" />
                      Remove broken branch to prevent potential failure
                    </li>
                    <li className="flex items-start">
                      <ChevronRight className="h-5 w-5 mr-2 flex-shrink-0" />
                      Assess structural integrity of remaining branches
                    </li>
                  </ul>
                </div>
                <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <h4 className="font-medium text-yellow-700 mb-2">Monitoring Suggested</h4>
                  <ul className="space-y-2 text-yellow-600">
                    <li className="flex items-start">
                      <ChevronRight className="h-5 w-5 mr-2 flex-shrink-0" />
                      Quarterly inspections of remaining structure
                    </li>
                    <li className="flex items-start">
                      <ChevronRight className="h-5 w-5 mr-2 flex-shrink-0" />
                      Monitor growth pattern near structure
                    </li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );

  return (
    <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg w-11/12 h-5/6 max-w-7xl flex flex-col overflow-hidden shadow-2xl">
        <div className="flex justify-between items-center p-4 border-b bg-slate-50">
          <h2 className="text-xl font-semibold flex items-center text-slate-800">
            <CheckCircle className="h-5 w-5 mr-2 text-green-600" />
            Tree Validation & Report
          </h2>
          <button 
            onClick={onClose} 
            className="p-2 rounded-full hover:bg-gray-200 text-gray-700"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        
        <div className="flex-1 overflow-auto">
          {showReport ? <ReportPreview /> : <ValidationInterface />}
        </div>
      </div>
    </div>
  );
};

export default ValidationSystem;