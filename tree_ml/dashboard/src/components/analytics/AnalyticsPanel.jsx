// src/components/analytics/AnalyticsPanel.jsx

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '../ui/card';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { 
  AlertTriangle, 
  ArrowRight, 
  Award, 
  BarChart,
  Brain,
  CheckCircle, 
  ChevronRight, 
  CircleAlert,
  FileCheck, 
  FileText, 
  Info, 
  Layers, 
  Map, 
  Ruler, 
  Save, 
  Search,
  Smartphone,
  Sparkles,
  TreePine,
  X,
  Activity,
  Heart,
  Leaf
} from 'lucide-react';
import { useGeminiTreeDetection } from '../../hooks/useGeminiTreeDetection';

/**
 * AnalyticsPanel - Full screen workflow for comprehensive tree analytics
 * This component displays detailed analytics across three categories:
 * Risk Assessment, Health Analysis, and Species Identification
 */
const AnalyticsPanel = ({
  treeData = null,
  selectedObject = null,
  onSaveToQueue = () => {},
  className = ""
}) => {
  const [activeTab, setActiveTab] = useState('risk');
  const [detectedObjects, setDetectedObjects] = useState([]);
  // Risk factors based on Tree Care requirements from the specs
  const [selectedRiskFactors, setSelectedRiskFactors] = useState({
    treeHealth: false,        // Dead, diseased, or structurally compromised
    structuralDefects: false, // Cracked, broken branches, or hanging limbs
    leaningAngle: false,      // Leaning angle (15-25°, 25-40°, 40°+)
    treeSize: false,          // Size categories (Small: 1-15ft, Medium: 15-30ft, Large: 30-100ft)
    proximityToStructures: false, // Trees near high-risk areas (homes, power lines, railroads, roads)
    canopyDensity: false,     // Tree canopy density issues
    deadLimbs: false,         // Dead or hazardous limbs
    rootSoilIssues: false     // Root and soil condition issues
  });
  const [riskLevel, setRiskLevel] = useState('no_risk');
  const [currentObject, setCurrentObject] = useState(null);
  const [customQuery, setCustomQuery] = useState('');
  
  // Use the Gemini tree detection hook
  const { 
    analyzeTree, 
    isAnalyzing, 
    analysisResult, 
    analysisError, 
    isGeminiEnabled 
  } = useGeminiTreeDetection();
  
  // Initialize or update current object when selected object changes
  useEffect(() => {
    if (selectedObject) {
      setCurrentObject(selectedObject);
    } else if (detectedObjects.length > 0 && !currentObject) {
      setCurrentObject(detectedObjects[0]);
    }
  }, [selectedObject, detectedObjects, currentObject]);
  
  // Get validation queue data from props or service
  useEffect(() => {
    // Format based on the validation_queue.json structure in backend mock data
    setDetectedObjects([
      { 
        id: 'val-001',
        tree_id: 'tree-001',
        property_id: 'prop-001',
        tree_species: 'Live Oak',
        tree_height: 45,
        tree_diameter: 24,
        tree_condition: 'Mature',
        canopy_density: 'Moderate',
        leaning_angle: '5° SE',
        lidar_height: 46.5,
        proximity: 'Near power lines',
        property_type: 'Residential',
        location: {
          description: '2082 Hillcrest Avenue, Dallas, TX 75230',
          coordinates: [-96.80542964970164, 32.87769319534349]
        },
        status: 'pending',
        coordinates: [-96.80542964970164, 32.87769319534349],
        risk_level: 'high',
        detection_source: 'LiDAR Detection',
        confidence: 0.92,
        analysis_status: 'pending',
        in_validation_queue: true,
        riskFactors: [
          {
            type: 'split_branch',
            level: 'high',
            description: 'Large split limb (15+ ft) detected'
          },
          {
            type: 'structure_proximity',
            level: 'medium',
            description: 'Canopy overhanging structure'
          }
        ],
        address: '2082 Hillcrest Avenue, Dallas, TX 75230'
      },
      { 
        id: 'val-002',
        tree_id: 'tree-003',
        property_id: 'prop-002',
        tree_species: 'Bald Cypress',
        tree_height: 60,
        tree_diameter: 30,
        tree_condition: 'Stressed',
        canopy_density: 'Sparse',
        leaning_angle: '18° NW',
        lidar_height: 61.2,
        proximity: 'Near roadway',
        property_type: 'Commercial',
        location: {
          description: '6121 Preston Hollow Road, Dallas, TX 75230',
          coordinates: [-96.755, 32.8937]
        },
        status: 'pending',
        coordinates: [-96.755, 32.8937],
        risk_level: 'medium',
        detection_source: 'ML Detection',
        confidence: 0.87,
        analysis_status: 'complete',
        in_validation_queue: true,
        riskFactors: [
          {
            type: 'lean',
            level: 'medium',
            description: 'Tree leaning 15-25 degrees'
          },
          {
            type: 'root_exposure',
            level: 'medium',
            description: 'Root system partially exposed'
          }
        ],
        address: '6121 Preston Hollow Road, Dallas, TX 75230'
      },
      { 
        id: 'val-003',
        tree_id: 'tree-002',
        property_id: 'prop-001',
        tree_species: 'Chinese Pistache',
        tree_height: 35,
        tree_diameter: 18,
        tree_condition: 'Healthy',
        canopy_density: 'Dense',
        leaning_angle: 'No Lean',
        lidar_height: 34.8,
        proximity: 'Near structure',
        property_type: 'Residential',
        location: {
          description: '2082 Hillcrest Avenue, Dallas, TX 75230',
          coordinates: [-96.80607053808008, 32.87694806481281]
        },
        status: 'pending',
        coordinates: [-96.80607053808008, 32.87694806481281],
        risk_level: 'low',
        detection_source: 'Manual Placement',
        confidence: 0.98,
        analysis_status: 'pending',
        in_validation_queue: true,
        riskFactors: [
          {
            type: 'foliage_discoloration',
            level: 'low',
            description: 'Minor canopy discoloration detected'
          }
        ],
        address: '2082 Hillcrest Avenue, Dallas, TX 75230'
      },
      { 
        id: 'val-004',
        tree_id: 'tree-005',
        property_id: 'prop-004',
        tree_species: 'Southern Magnolia',
        tree_height: 38,
        tree_diameter: 20,
        tree_condition: 'Diseased',
        canopy_density: 'Dense',
        leaning_angle: 'No Lean',
        lidar_height: 37.5,
        proximity: 'Near fence',
        property_type: 'Residential',
        location: {
          description: '8350 Forest Lane, Dallas, TX 75243',
          coordinates: [-96.7595, 32.9075]
        },
        status: 'pending',
        coordinates: [-96.7595, 32.9075],
        risk_level: 'no_risk',
        detection_source: 'ML Detection',
        confidence: 0.94,
        analysis_status: 'pending',
        in_validation_queue: true,
        riskFactors: [],
        address: '8350 Forest Lane, Dallas, TX 75243'
      }
    ]);
  }, []);
  
  const handleRiskFactorChange = (factor) => {
    // Toggle the selected risk factor
    setSelectedRiskFactors(prev => ({
      ...prev,
      [factor]: !prev[factor]
    }));
    
    // Create new risk factors state to evaluate
    const newRiskFactors = {
      ...selectedRiskFactors,
      [factor]: !selectedRiskFactors[factor]
    };
    
    // Get count of selected factors
    const factorCount = Object.values(newRiskFactors).filter(Boolean).length;
    
    // Define critical factors based on tree care requirements
    const criticalFactors = ['treeHealth', 'structuralDefects', 'deadLimbs', 'proximityToStructures'];
    
    // Check if any critical factor is selected
    const hasCriticalFactor = criticalFactors.some(f => newRiskFactors[f]);
    
    // Update risk level based on selected factors
    if (factorCount >= 3 || hasCriticalFactor) {
      setRiskLevel('high');
    } else if (factorCount >= 1) {
      setRiskLevel('medium');
    } else if (factorCount === 0 && !hasCriticalFactor) {
      setRiskLevel('low');
    } else {
      setRiskLevel('no_risk');
    }
  };
  
  const saveToValidationQueue = () => {
    if (!currentObject) return;
    
    // Define critical factors based on tree care requirements
    const criticalFactors = ['treeHealth', 'structuralDefects', 'deadLimbs', 'proximityToStructures'];
    
    // Convert selected risk factors to the format used in validation_queue.json
    const riskFactorsArray = Object.entries(selectedRiskFactors)
      .filter(([_, selected]) => selected)
      .map(([factor]) => {
        const isCritical = criticalFactors.includes(factor);
        const level = isCritical ? 'high' : 'medium';
        
        // Convert the factor name to snake_case for type field
        const typeMap = {
          'treeHealth': 'tree_health',
          'structuralDefects': 'structural_defect',
          'leaningAngle': 'lean',
          'treeSize': 'size',
          'proximityToStructures': 'structure_proximity',
          'canopyDensity': 'canopy_density',
          'deadLimbs': 'deadwood',
          'rootSoilIssues': 'root_exposure'
        };
        
        return {
          type: typeMap[factor] || factor.toLowerCase(),
          level: level,
          description: getFactorDescription(factor)
        };
      });
    
    const updatedObject = {
      ...currentObject,
      risk_level: riskLevel,
      riskFactors: riskFactorsArray,
      status: 'pending',
      analysis_status: 'complete',
      // Add Gemini analysis status if available
      gemini_analysis: currentObject.gemini_analysis || null
    };
    
    onSaveToQueue(updatedObject);
  };
  
  // Helper function to get human-readable descriptions for factors
  const getFactorDescription = (factor) => {
    const descriptions = {
      treeHealth: 'Dead, diseased, or structurally compromised',
      structuralDefects: 'Cracked, broken branches, or hanging limbs',
      leaningAngle: 'Leaning angle (15-25°, 25-40°, 40°+)',
      treeSize: 'Size categories (Small: 1-15ft, Medium: 15-30ft, Large: 30-100ft)',
      proximityToStructures: 'Trees near high-risk areas (homes, power lines, railroads, roads)',
      canopyDensity: 'Tree canopy density issues',
      deadLimbs: 'Dead or hazardous limbs',
      rootSoilIssues: 'Root and soil condition issues'
    };
    
    return descriptions[factor] || factor;
  };
  
  // Count selected risk factors
  const getSelectedFactorCount = () => {
    return Object.values(selectedRiskFactors).filter(Boolean).length;
  };
  
  // Generate query template based on selected risk factors
  const generateQueryTemplate = () => {
    if (!currentObject) return '';
    
    const factorDescriptions = {
      treeHealth: "tree health issues (dead, diseased, or compromised)",
      structuralDefects: "structural defects (cracked branches, hanging limbs)",
      leaningAngle: "leaning angle severity",
      treeSize: "tree size concerns (height/width implications)",
      proximityToStructures: "proximity to structures (buildings, power lines, roads)",
      canopyDensity: "canopy density issues",
      deadLimbs: "dead or hazardous limbs",
      rootSoilIssues: "root system and soil condition issues"
    };
    
    // Get selected factors
    const selectedFactors = Object.entries(selectedRiskFactors)
      .filter(([_, selected]) => selected)
      .map(([factor, _]) => factorDescriptions[factor] || factor);
    
    // Extract coordinates in the right format
    const coords = currentObject.location && currentObject.location.coordinates ? 
      currentObject.location.coordinates.join(', ') : 
      (currentObject.coordinates ? currentObject.coordinates.join(', ') : 'Unknown');
    
    // Tree details
    const treeDetails = [
      `ID: ${currentObject.id}`,
      `Tree ID: ${currentObject.tree_id || currentObject.id}`,
      `Species: ${currentObject.tree_species || 'Unknown'}`,
      `Condition: ${currentObject.tree_condition || 'Unknown'}`,
      `Height: ${currentObject.tree_height || 'Unknown'} ft`,
      `Diameter: ${currentObject.tree_diameter || 'Unknown'} in`,
      `Canopy Density: ${currentObject.canopy_density || 'Unknown'}`,
      `Leaning Angle: ${currentObject.leaning_angle || 'None'}`,
      `Proximity: ${currentObject.proximity || 'Unknown'}`,
      `Address: ${currentObject.address || (currentObject.location ? currentObject.location.description : 'Unknown')}`,
      `Coordinates: [${coords}]`
    ].join('\n');
    
    // Generate the query template
    let query = `Analyze the following tree for risk assessment purposes:\n\n${treeDetails}\n\n`;
    
    if (selectedFactors.length > 0) {
      query += `Please focus on analyzing these specific risk factors:\n`;
      selectedFactors.forEach(factor => {
        query += `- ${factor}\n`;
      });
      query += '\n';
    }
    
    query += `Provide the following in your analysis:
1. Risk Summary: A brief assessment of the overall risk level
2. Detailed Analysis: Specific insights about each risk factor
3. Recommendations: Suggested actions to mitigate risks
4. Future Concerns: Potential issues that may develop over time

For tree care professionals, please use arborist terminology where appropriate.`;
    
    return query;
  };
  
  // Update the query template when risk factors change
  useEffect(() => {
    if (currentObject && (activeTab === 'analysis' || isAnalyzing)) {
      setCustomQuery(generateQueryTemplate());
    }
  }, [selectedRiskFactors, currentObject, activeTab]);
  
  // Handle Gemini Analysis for the current object
  const handleGeminiAnalysis = async () => {
    if (!currentObject || !isGeminiEnabled) return;
    
    try {
      // Use the custom query if available, otherwise generate a new one
      const queryToUse = customQuery || generateQueryTemplate();
      
      // Call the Gemini analysis hook with the custom query
      const result = await analyzeTree(currentObject, queryToUse);
      
      if (result?.success && result?.structured_analysis) {
        // If successful, update the current object with Gemini analysis results
        setCurrentObject(prev => ({
          ...prev,
          gemini_analysis: result.structured_analysis,
          analysis_timestamp: new Date().toISOString()
        }));
        
        // Use Gemini's analysis to suggest risk factors
        if (result.suggested_risk_factors) {
          // This would be populated by the backend Gemini analysis
          const suggestedFactors = result.suggested_risk_factors;
          
          // Update selected risk factors based on suggestions
          setSelectedRiskFactors(prev => ({
            ...prev,
            ...suggestedFactors
          }));
          
          // Re-calculate risk level
          const factorCount = Object.values(suggestedFactors).filter(Boolean).length;
          const criticalFactors = ['treeHealth', 'structuralDefects', 'deadLimbs', 'proximityToStructures'];
          const hasCriticalFactor = criticalFactors.some(f => suggestedFactors[f]);
          
          if (factorCount >= 3 || hasCriticalFactor) {
            setRiskLevel('high');
          } else if (factorCount >= 1) {
            setRiskLevel('medium');
          }
        }
      }
    } catch (error) {
      console.error('Error analyzing with Gemini:', error);
    }
  };
  
  // Function to close the panel
  const closePanel = () => {
    window.dispatchEvent(new CustomEvent('closeAnalyticsPanel', {
      detail: { source: 'close_button' }
    }));
  };

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40 animate-fadeIn"
      onClick={closePanel} // Close when clicking the backdrop
    >
      <div 
        className="bg-white rounded-md shadow-xl overflow-hidden w-5/6 max-w-5xl h-5/6 flex flex-col animate-scaleIn"
        onClick={(e) => e.stopPropagation()} // Prevent closing when clicking the panel itself
      >
        {/* Header with Navigation tabs - Dark Emerald Green Theme */}
        <div className="bg-emerald-950 px-4 pt-2 pb-0 shadow-md">
          <div className="flex justify-between border-b border-emerald-800">
            <div className="flex">
              <button 
                className={`px-4 py-2 text-sm font-medium flex items-center ${
                  activeTab === 'risk' 
                    ? 'border-b-2 border-emerald-300 text-white bg-emerald-900' 
                    : 'text-emerald-200 hover:text-white hover:bg-emerald-900/50'
                }`}
                onClick={() => setActiveTab('risk')}
              >
                <AlertTriangle className="h-4 w-4 mr-2" />
                Risk
              </button>
              <button 
                className={`px-4 py-2 text-sm font-medium flex items-center ${
                  activeTab === 'health' 
                    ? 'border-b-2 border-emerald-300 text-white bg-emerald-900' 
                    : 'text-emerald-200 hover:text-white hover:bg-emerald-900/50'
                }`}
                onClick={() => setActiveTab('health')}
              >
                <Heart className="h-4 w-4 mr-2" />
                Health
              </button>
              <button 
                className={`px-4 py-2 text-sm font-medium flex items-center ${
                  activeTab === 'species' 
                    ? 'border-b-2 border-emerald-300 text-white bg-emerald-900' 
                    : 'text-emerald-200 hover:text-white hover:bg-emerald-900/50'
                }`}
                onClick={() => setActiveTab('species')}
              >
                <Leaf className="h-4 w-4 mr-2" />
                Species
              </button>
            </div>
            <button 
              className="text-emerald-200 hover:text-white focus:outline-none p-1 flex items-center mr-1"
              onClick={closePanel}
              title="Close analytics panel"
            >
              <X size={16} />
            </button>
          </div>
        </div>
        
        {/* Content area - Scrollable with padding */}
        <div className="flex-1 overflow-auto bg-gray-50">
        {activeTab === 'risk' && (
          <div className="space-y-3 p-4">
            {!currentObject ? (
              <Card className="border-slate-200">
                <CardContent className="p-4 text-center">
                  <Search className="h-8 w-8 mx-auto mb-2 text-slate-400" />
                  <p className="text-sm text-slate-600 font-medium">
                    No object selected
                  </p>
                  <p className="text-xs text-slate-500 mt-1">
                    Select an object on the map or from the detection sidebar
                  </p>
                </CardContent>
              </Card>
            ) : (
              <>
                {/* Horizontal Validation Queue (Object Counter) */}
                <Card className="mb-3 border-0 rounded-md shadow-[0_1px_3px_0_rgba(6,78,59,0.1)]">
                  <CardHeader className="pb-2 pt-3 px-3 bg-emerald-900">
                    <CardTitle className="text-xs text-white uppercase tracking-wider flex items-center justify-between">
                      <span>Objects for Validation ({detectedObjects.length})</span>
                      <button 
                        className="text-xs text-emerald-100 hover:text-white flex items-center"
                        onClick={() => {
                          // Navigate to validation queue panel
                          console.log('Navigate to validation queue');
                          
                          // Trigger the validation queue panel to open
                          window.dispatchEvent(new CustomEvent('openReviewPanel', {
                            detail: { 
                              source: 'analyticsPanel' 
                            }
                          }));
                        }}
                      >
                        View All
                        <ChevronRight className="h-3 w-3 ml-0.5" />
                      </button>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="px-3 py-3">
                    <div className="relative pb-1">
                      <div className="overflow-auto flex space-x-2 scrollbar-none max-w-full h-[68px]">
                        {detectedObjects.map(object => (
                          <div 
                            key={object.id} 
                            className={`flex-shrink-0 w-28 rounded-md border cursor-pointer transition ${
                              object.id === (currentObject?.id) 
                                ? 'border-emerald-900 bg-gray-50' 
                                : 'border-slate-200 hover:border-emerald-300 hover:bg-gray-50/30'
                            }`}
                            onClick={() => setCurrentObject(object)}
                          >
                            <div className="p-1.5">
                              <div className="flex items-center justify-between mb-0.5">
                                <div className={`w-2 h-2 rounded-full ${
                                  object.risk_level === 'high' ? 'bg-red-500' :
                                  object.risk_level === 'medium' ? 'bg-orange-500' :
                                  object.risk_level === 'low' ? 'bg-yellow-500' :
                                  'bg-green-500'
                                }`}></div>
                                <div className={`text-[8px] py-0.5 px-1 rounded ${
                                  object.analysis_status === 'complete' ? 'bg-green-50 text-green-700' : 'bg-amber-50 text-amber-700'
                                }`}>
                                  {object.analysis_status === 'complete' ? 'Done' : 'Pending'}
                                </div>
                              </div>
                              <div className="text-xs font-medium text-slate-700 truncate">
                                {object.tree_species || object.species || 'Unspecified'}
                              </div>
                              <div className="text-[8px] text-slate-500 truncate">
                                {object.id}
                              </div>
                            </div>
                          </div>
                        ))}
                        
                        {detectedObjects.length === 0 && (
                          <div className="flex items-center justify-center w-full py-3 px-2 text-center">
                            <p className="text-xs text-slate-500">
                              No objects in validation queue
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Object quick status */}
                <div className="flex items-center justify-between px-1">
                  <div className="flex items-center">
                    <TreePine className="h-4 w-4 mr-1.5 text-slate-700" />
                    <span className="text-sm font-medium text-slate-800">
                      {currentObject.tree_species || 'Unspecified'} {currentObject.tree_condition ? `(${currentObject.tree_condition})` : ''}
                    </span>
                  </div>
                  <div className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
                    currentObject.risk_level === 'high' ? 'bg-red-50 text-red-700' :
                    currentObject.risk_level === 'medium' ? 'bg-orange-50 text-orange-700' :
                    currentObject.risk_level === 'low' ? 'bg-yellow-100 text-yellow-700' :
                    'bg-green-50 text-green-700'
                  }`}>
                    {currentObject.risk_level === 'high' ? 'High Risk' :
                     currentObject.risk_level === 'medium' ? 'Medium Risk' :
                     currentObject.risk_level === 'low' ? 'Low Risk' :
                     'No Risk'}
                  </div>
                </div>
                
                {/* Basic object details */}
                <Card className="border-0 rounded-md overflow-hidden shadow-[0_1px_3px_0_rgba(6,78,59,0.1)]">
                  <CardHeader className="pb-1.5 pt-2 px-3 bg-emerald-900 rounded-none">
                    <CardTitle className="text-xs text-white uppercase tracking-wider">
                      Object Details
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="px-3 pb-3 pt-2 grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <div className="text-xs text-slate-500 mb-0.5">ID</div>
                      <div className="font-mono text-xs text-slate-700">{currentObject.id}</div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 mb-0.5">Source</div>
                      <div className="flex items-center">
                        <span className="text-xs text-slate-700">{currentObject.detection_source}</span>
                        {currentObject.detection_source === 'ML Detection' && (
                          <span className="ml-1 px-1 py-0.5 bg-blue-50 text-blue-700 rounded text-[10px]">
                            {Math.round(currentObject.confidence * 100)}%
                          </span>
                        )}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 mb-0.5">Name / Species</div>
                      <div className="flex items-center">
                        <span className="text-xs text-slate-700">
                          {currentObject.tree_species ? `${currentObject.tree_species}` : 'Unknown'}
                        </span>
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 mb-0.5">Address</div>
                      <div className="flex items-center">
                        <span className="text-xs text-slate-700">
                          {currentObject.address ? currentObject.address : '123 Example St'}
                        </span>
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 mb-0.5">Height</div>
                      <div className="flex items-center">
                        <span className="text-xs text-slate-700">
                          {currentObject.tree_height ? `${currentObject.tree_height} ft` : 'Unknown'}
                        </span>
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 mb-0.5">Diameter</div>
                      <div className="flex items-center">
                        <span className="text-xs text-slate-700">
                          {currentObject.tree_diameter ? `${currentObject.tree_diameter} in` : 'Unknown'}
                        </span>
                      </div>
                    </div>
                    <div className="col-span-2">
                      <div className="text-xs text-slate-500 mb-0.5">Coordinates</div>
                      <div className="flex items-center">
                        <span className="text-xs font-mono text-slate-700">
                          {currentObject.location && currentObject.location.coordinates ? 
                            `[${currentObject.location.coordinates[0].toFixed(6)}, ${currentObject.location.coordinates[1].toFixed(6)}]` : 
                            'Unknown'}
                        </span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                {/* Analysis & risk status */}
                <Card className="border-0 shadow-sm mt-3">
                  <CardHeader className="pb-1.5 pt-2 px-3 bg-emerald-900 rounded-none">
                    <CardTitle className="text-xs text-white uppercase tracking-wider flex items-center justify-between">
                      <span>Risk Analysis</span>
                      {currentObject.analysis_status === 'complete' ? (
                        <span className="bg-green-50 text-green-700 px-1.5 py-0.5 rounded text-[10px] font-medium">Complete</span>
                      ) : (
                        <span className="bg-amber-50 text-amber-700 px-1.5 py-0.5 rounded text-[10px] font-medium">Needs Analysis</span>
                      )}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="px-3 pb-3 pt-3">
                    {currentObject.analysis_status === 'complete' ? (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-slate-700">Risk Level</span>
                          <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
                            currentObject.risk_level === 'high' ? 'bg-red-50 text-red-700' :
                            currentObject.risk_level === 'medium' ? 'bg-orange-50 text-orange-700' :
                            'bg-green-50 text-green-700'
                          }`}>
                            {currentObject.risk_level === 'high' ? 'High Risk' :
                             currentObject.risk_level === 'medium' ? 'Medium Risk' :
                             'Low Risk'}
                          </span>
                        </div>
                        
                        {currentObject.risk_factors && currentObject.risk_factors.length > 0 ? (
                          <div className="mt-2">
                            <div className="text-xs text-slate-600 mb-1">Risk Factors:</div>
                            <div className="space-y-1">
                              {currentObject.risk_factors.map(factor => (
                                <div key={factor.factor} className="flex items-center">
                                  <AlertTriangle className="h-3 w-3 mr-1 text-amber-500" />
                                  <span className="text-xs text-slate-700 capitalize">{factor.factor}</span>
                                  <span className={`ml-auto text-[10px] px-1.5 rounded-full ${
                                    factor.level === 'high' ? 'bg-red-50 text-red-700' : 
                                    'bg-orange-50 text-orange-700'
                                  }`}>
                                    {factor.level}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        ) : (
                          <div className="flex items-center justify-center mt-2 bg-slate-50 p-2 rounded">
                            <CheckCircle className="h-3.5 w-3.5 mr-1 text-green-600" />
                            <span className="text-xs text-slate-700">No significant risk factors</span>
                          </div>
                        )}
                        
                        <div className="flex space-x-2 mt-3">
                          <Button 
                            className="flex-1 bg-emerald-800 hover:bg-emerald-700 text-white shadow-[0_1px_2px_0_rgba(6,78,59,0.1)] text-sm py-1.5 h-auto rounded-md"
                            onClick={() => setActiveTab('health')}
                          >
                            <CircleAlert className="h-3.5 w-3.5 mr-1.5" />
                            Edit Analysis
                          </Button>
                          <Button 
                            className="flex-1 bg-white hover:bg-emerald-50 text-emerald-800 text-sm px-2 py-1.5 h-auto rounded-md border border-emerald-300"
                            onClick={() => saveToValidationQueue()}
                          >
                            <Save className="h-3.5 w-3.5 mr-1.5" />
                            Save to Queue
                          </Button>
                        </div>
                        
                      </div>
                    ) : (
                      <div className="flex flex-col items-center justify-center py-2">
                        <div className="bg-amber-50 p-2 rounded-md mb-2 w-full">
                          <p className="text-xs text-amber-700 text-center flex items-center justify-center">
                            <AlertTriangle className="h-3.5 w-3.5 mr-1.5 text-amber-600" />
                            This object requires risk analysis
                          </p>
                        </div>
                        
                        <div className="mt-4 grid grid-cols-2 gap-2">
                          <Button 
                            className="flex-1 bg-emerald-800 hover:bg-emerald-700 text-white shadow-[0_1px_2px_0_rgba(6,78,59,0.1)] text-sm py-2 h-auto rounded-md border border-emerald-600"
                            onClick={() => setActiveTab('health')}
                          >
                            <CircleAlert className="h-3.5 w-3.5 mr-1.5" />
                            Start Analysis
                          </Button>
                          
                          <Button 
                            className="flex-1 bg-white hover:bg-emerald-50 text-emerald-800 font-semibold text-sm py-2 h-auto rounded-md border border-emerald-300"
                            onClick={() => {
                              // Display a dialog or modal to review the prompt
                              alert("Review Prompt: This would show the current analysis prompt for review.");
                            }}
                          >
                            <FileCheck className="h-3.5 w-3.5 mr-1.5" />
                            Review Prompt
                          </Button>
                        </div>
                        
                      </div>
                    )}
                  </CardContent>
                </Card>
                
                {/* Risk Level Guidelines section moved from Health tab to Risk tab */}
                <Card className="border-0 rounded-md mt-3 shadow-[0_1px_3px_0_rgba(6,78,59,0.1)]">
                  <CardHeader className="pb-1.5 pt-2 px-3 bg-emerald-900 rounded-none">
                    <CardTitle className="text-xs text-white uppercase tracking-wider">
                      Risk Level Guidelines
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="px-3 pb-3 pt-3 text-xs space-y-1.5">
                    <div className="flex items-center">
                      <div className="w-2 h-2 rounded-full bg-red-500 mr-2"></div>
                      <span className="font-medium text-slate-700">High Risk</span>
                      <span className="ml-2 text-slate-600">3+ factors or critical factors</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-2 h-2 rounded-full bg-orange-500 mr-2"></div>
                      <span className="font-medium text-slate-700">Medium Risk</span>
                      <span className="ml-2 text-slate-600">1-2 risk factors</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-2 h-2 rounded-full bg-yellow-500 mr-2"></div>
                      <span className="font-medium text-slate-700">Low Risk</span>
                      <span className="ml-2 text-slate-600">1-2 non-critical factors</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-2 h-2 rounded-full bg-green-500 mr-2"></div>
                      <span className="font-medium text-slate-700">No Risk</span>
                      <span className="ml-2 text-slate-600">Default new status</span>
                    </div>
                    <div className="mt-2 pt-1 border-t border-slate-100">
                      <div className="text-slate-600 text-[10px] italic">
                        Critical factors: Tree Health, Structural Defects, Dead Limbs, Proximity to Structures
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                {/* Gemini Analysis Card - shown if analysis has been run */}
                {currentObject?.gemini_analysis && (
                  <Card className="border-0 rounded-md mt-3 shadow-[0_1px_3px_0_rgba(6,78,59,0.1)]">
                    <CardHeader className="pb-1.5 pt-2 px-3 bg-emerald-900 rounded-none">
                      <CardTitle className="text-xs text-white uppercase tracking-wider flex items-center">
                        <Brain className="h-3.5 w-3.5 mr-1.5 text-white" />
                        Gemini AI Analysis
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="px-3 pb-3 pt-3">
                      <div className="text-xs text-slate-700 space-y-2">
                        {currentObject.gemini_analysis.risk_summary && (
                          <div>
                            <p className="font-medium mb-1">Risk Summary:</p>
                            <p className="text-slate-600 bg-gray-50 p-1.5 rounded text-[11px]">
                              {currentObject.gemini_analysis.risk_summary}
                            </p>
                          </div>
                        )}
                        {currentObject.gemini_analysis.recommendations && (
                          <div>
                            <p className="font-medium mb-1">Recommendations:</p>
                            <p className="text-slate-600 bg-green-50 p-1.5 rounded text-[11px]">
                              {currentObject.gemini_analysis.recommendations}
                            </p>
                          </div>
                        )}
                        {currentObject.analysis_timestamp && (
                          <p className="text-[10px] text-slate-500 mt-1 italic">
                            Analysis performed: {new Date(currentObject.analysis_timestamp).toLocaleString()}
                          </p>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                )}
                
                {/* End of Risk tab content */}
              </>
            )}
          </div>
        )}
        
        {activeTab === 'health' && (
          <div className="space-y-3 p-4">
            {!currentObject ? (
              <Card className="border-slate-200">
                <CardContent className="p-4 text-center">
                  <Search className="h-8 w-8 mx-auto mb-2 text-slate-400" />
                  <p className="text-sm text-slate-600 font-medium">
                    No object selected
                  </p>
                  <p className="text-xs text-slate-500 mt-1">
                    Select an object on the map or from the detection sidebar
                  </p>
                </CardContent>
              </Card>
            ) : (
              <>
                {/* Horizontal Validation Queue (Object Counter) */}
                <Card className="mb-3 border-0 rounded-md shadow-[0_1px_3px_0_rgba(6,78,59,0.1)]">
                  <CardHeader className="pb-1.5 pt-2 px-3 bg-emerald-900 rounded-none">
                    <CardTitle className="text-xs text-white uppercase tracking-wider flex items-center justify-between">
                      <span>Objects for Validation ({detectedObjects.length})</span>
                      <button 
                        className="text-xs text-emerald-100 hover:text-white flex items-center"
                        onClick={() => {
                          // Navigate to validation queue panel
                          console.log('Navigate to validation queue');
                          
                          // Trigger the validation queue panel to open
                          window.dispatchEvent(new CustomEvent('openReviewPanel', {
                            detail: { 
                              source: 'analyticsPanel' 
                            }
                          }));
                        }}
                      >
                        View All
                        <ChevronRight className="h-3 w-3 ml-0.5" />
                      </button>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="px-3 py-3">
                    <div className="relative pb-1">
                      <div className="overflow-auto flex space-x-2 scrollbar-none max-w-full h-[68px]">
                        {detectedObjects.map(object => (
                          <div 
                            key={object.id} 
                            className={`flex-shrink-0 w-28 rounded-md border cursor-pointer transition ${
                              object.id === (currentObject?.id) 
                                ? 'border-emerald-900 bg-gray-50' 
                                : 'border-slate-200 hover:border-emerald-300 hover:bg-gray-50/30'
                            }`}
                            onClick={() => setCurrentObject(object)}
                          >
                            <div className="p-1.5">
                              <div className="flex items-center justify-between mb-0.5">
                                <div className={`w-2 h-2 rounded-full ${
                                  object.risk_level === 'high' ? 'bg-red-500' :
                                  object.risk_level === 'medium' ? 'bg-orange-500' :
                                  object.risk_level === 'low' ? 'bg-yellow-500' :
                                  'bg-green-500'
                                }`}></div>
                                <div className={`text-[8px] py-0.5 px-1 rounded ${
                                  object.analysis_status === 'complete' ? 'bg-green-50 text-green-700' : 'bg-amber-50 text-amber-700'
                                }`}>
                                  {object.analysis_status === 'complete' ? 'Done' : 'Pending'}
                                </div>
                              </div>
                              <div className="text-xs font-medium text-slate-700 truncate">
                                {object.tree_species || object.species || 'Unspecified'}
                              </div>
                              <div className="text-[8px] text-slate-500 truncate">
                                {object.id}
                              </div>
                            </div>
                          </div>
                        ))}
                        
                        {detectedObjects.length === 0 && (
                          <div className="flex items-center justify-center w-full py-3 px-2 text-center">
                            <p className="text-xs text-slate-500">
                              No objects in validation queue
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                {/* Object Details */}
                <Card className="border-0 rounded-md overflow-hidden shadow-[0_1px_3px_0_rgba(6,78,59,0.1)]">
                  <CardHeader className="pb-1.5 pt-2 px-3 bg-emerald-900 rounded-none">
                    <CardTitle className="text-xs text-white uppercase tracking-wider">
                      Object Details
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="px-3 pb-3 pt-3 grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <div className="text-xs text-slate-500 mb-0.5">ID</div>
                      <div className="font-mono text-xs text-slate-700">{currentObject.id}</div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 mb-0.5">Source</div>
                      <div className="flex items-center">
                        <span className="text-xs text-slate-700">{currentObject.detection_source}</span>
                        {currentObject.detection_source === 'ML Detection' && (
                          <span className="ml-1 px-1 py-0.5 bg-blue-50 text-blue-700 rounded text-[10px]">
                            {Math.round(currentObject.confidence * 100)}%
                          </span>
                        )}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 mb-0.5">Name / Species</div>
                      <div className="flex items-center">
                        <span className="text-xs text-slate-700">
                          {currentObject.tree_species ? `${currentObject.tree_species}` : 'Unknown'}
                        </span>
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 mb-0.5">Address</div>
                      <div className="flex items-center">
                        <span className="text-xs text-slate-700">
                          {currentObject.address ? currentObject.address : '123 Example St'}
                        </span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                <Card className="border-0 rounded-md shadow-[0_1px_3px_0_rgba(6,78,59,0.1)] mt-3">
                  <CardHeader className="pb-1.5 pt-2 px-3 bg-emerald-900 rounded-none">
                    <CardTitle className="text-xs text-white uppercase tracking-wider flex items-center">
                      <Heart className="h-3.5 w-3.5 mr-1.5" />
                      Health Assessment
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="px-3 pb-3 pt-3">
                    <div className="space-y-4">
                      <div className="space-y-1.5">
                        <label className="text-xs text-slate-700 font-medium">Tree Health Status</label>
                        <div className="flex items-center rounded-md p-1.5 bg-slate-50 border border-slate-200">
                          <div className="flex-1 flex items-center">
                            <span className="w-2 h-2 rounded-full mr-2 bg-emerald-500"></span>
                            <span className="text-xs font-medium text-slate-800">
                              {currentObject.tree_condition || 'Unknown'}
                            </span>
                          </div>
                          <div className="text-[10px] text-slate-600">
                            Last assessed: {new Date().toLocaleDateString()}
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-2">
                        <label className="text-xs text-slate-700 font-medium">Health Indicators</label>
                        <div className="space-y-1.5">
                          {/* Health indicators with slate theme to match risk assessment items */}
                          <div className="flex items-center justify-between p-1.5 rounded-md bg-slate-50 border border-slate-200">
                            <div className="flex items-center">
                              <Activity className="h-3.5 w-3.5 mr-1.5 text-slate-700" />
                              <span className="text-xs text-slate-800">Canopy Density</span>
                            </div>
                            <div className="text-xs font-medium text-slate-900">
                              {currentObject.canopy_density || 'Normal'}
                            </div>
                          </div>
                          
                          <div className="flex items-center justify-between p-1.5 rounded-md bg-slate-50 border border-slate-200">
                            <div className="flex items-center">
                              <Activity className="h-3.5 w-3.5 mr-1.5 text-slate-700" />
                              <span className="text-xs text-slate-800">Foliage Color</span>
                            </div>
                            <div className="text-xs font-medium text-slate-900">
                              Good
                            </div>
                          </div>
                          
                          <div className="flex items-center justify-between p-1.5 rounded-md bg-slate-50 border border-slate-200">
                            <div className="flex items-center">
                              <Activity className="h-3.5 w-3.5 mr-1.5 text-slate-700" />
                              <span className="text-xs text-slate-800">Trunk Condition</span>
                            </div>
                            <div className="text-xs font-medium text-slate-900">
                              Healthy
                            </div>
                          </div>
                          
                          <div className="flex items-center justify-between p-1.5 rounded-md bg-slate-50 border border-slate-200">
                            <div className="flex items-center">
                              <Activity className="h-3.5 w-3.5 mr-1.5 text-slate-700" />
                              <span className="text-xs text-slate-800">Growth Pattern</span>
                            </div>
                            <div className="text-xs font-medium text-slate-900">
                              Normal
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {/* Recommendations section */}
                      <div>
                        <label className="text-xs text-slate-700 font-medium">Health Recommendations</label>
                        <div className="mt-1 p-2 bg-white border border-slate-100 rounded-md text-xs text-slate-700">
                          <p>Based on the assessment, this tree appears to be in good health. Regular monitoring is recommended to maintain its condition.</p>
                        </div>
                      </div>
                      
                      {/* Add the buttons at the bottom of Health Assessment card */}
                      <div className="mt-4 grid grid-cols-2 gap-2">
                        <Button 
                          className="flex-1 bg-emerald-800 hover:bg-emerald-700 text-white shadow-[0_1px_2px_0_rgba(6,78,59,0.1)] text-sm py-2 h-auto rounded-md border border-emerald-600"
                          onClick={handleGeminiAnalysis}
                          disabled={isAnalyzing || !customQuery}
                        >
                          {isAnalyzing ? (
                            <>
                              <div className="h-3.5 w-3.5 mr-1.5 animate-spin border-2 border-white border-opacity-20 border-t-white rounded-full"></div>
                              Analyzing...
                            </>
                          ) : (
                            <>
                              <Sparkles className="h-3.5 w-3.5 mr-1.5" />
                              Full Assessment
                            </>
                          )}
                        </Button>
                        <Button 
                          className="flex-1 bg-white hover:bg-emerald-50 text-emerald-800 font-semibold text-sm py-2 h-auto rounded-md border border-emerald-300"
                          onClick={() => {
                            // Save health report
                            alert("Health report saved successfully");
                          }}
                        >
                          <Save className="h-3.5 w-3.5 mr-1.5" />
                          Health Report
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        )}
        
        {/* Species Tab */}
        {activeTab === 'species' && (
          <div className="space-y-3 p-4">
            {!currentObject ? (
              <Card className="border-slate-200">
                <CardContent className="p-4 text-center">
                  <Search className="h-8 w-8 mx-auto mb-2 text-slate-400" />
                  <p className="text-sm text-slate-600 font-medium">
                    No object selected
                  </p>
                  <p className="text-xs text-slate-500 mt-1">
                    Select an object on the map or from the detection sidebar
                  </p>
                </CardContent>
              </Card>
            ) : (
              <>
                {/* Horizontal Validation Queue (Object Counter) */}
                <Card className="mb-3 border-0 rounded-md shadow-[0_1px_3px_0_rgba(6,78,59,0.1)]">
                  <CardHeader className="pb-1.5 pt-2 px-3 bg-emerald-900 rounded-none">
                    <CardTitle className="text-xs text-white uppercase tracking-wider flex items-center justify-between">
                      <span>Objects for Validation ({detectedObjects.length})</span>
                      <button 
                        className="text-xs text-emerald-100 hover:text-white flex items-center"
                        onClick={() => {
                          // Navigate to validation queue panel
                          console.log('Navigate to validation queue');
                          
                          // Trigger the validation queue panel to open
                          window.dispatchEvent(new CustomEvent('openReviewPanel', {
                            detail: { 
                              source: 'analyticsPanel' 
                            }
                          }));
                        }}
                      >
                        View All
                        <ChevronRight className="h-3 w-3 ml-0.5" />
                      </button>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="px-3 py-3">
                    <div className="relative pb-1">
                      <div className="overflow-auto flex space-x-2 scrollbar-none max-w-full h-[68px]">
                        {detectedObjects.map(object => (
                          <div 
                            key={object.id} 
                            className={`flex-shrink-0 w-28 rounded-md border cursor-pointer transition ${
                              object.id === (currentObject?.id) 
                                ? 'border-emerald-900 bg-gray-50' 
                                : 'border-slate-200 hover:border-emerald-300 hover:bg-gray-50/30'
                            }`}
                            onClick={() => setCurrentObject(object)}
                          >
                            <div className="p-1.5">
                              <div className="flex items-center justify-between mb-0.5">
                                <div className={`w-2 h-2 rounded-full ${
                                  object.risk_level === 'high' ? 'bg-red-500' :
                                  object.risk_level === 'medium' ? 'bg-orange-500' :
                                  object.risk_level === 'low' ? 'bg-yellow-500' :
                                  'bg-green-500'
                                }`}></div>
                                <div className={`text-[8px] py-0.5 px-1 rounded ${
                                  object.analysis_status === 'complete' ? 'bg-green-50 text-green-700' : 'bg-amber-50 text-amber-700'
                                }`}>
                                  {object.analysis_status === 'complete' ? 'Done' : 'Pending'}
                                </div>
                              </div>
                              <div className="text-xs font-medium text-slate-700 truncate">
                                {object.tree_species || object.species || 'Unspecified'}
                              </div>
                              <div className="text-[8px] text-slate-500 truncate">
                                {object.id}
                              </div>
                            </div>
                          </div>
                        ))}
                        
                        {detectedObjects.length === 0 && (
                          <div className="flex items-center justify-center w-full py-3 px-2 text-center">
                            <p className="text-xs text-slate-500">
                              No objects in validation queue
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                {/* Object Details */}
                <Card className="border-0 rounded-md overflow-hidden shadow-[0_1px_3px_0_rgba(6,78,59,0.1)]">
                  <CardHeader className="pb-1.5 pt-2 px-3 bg-emerald-900 rounded-none">
                    <CardTitle className="text-xs text-white uppercase tracking-wider">
                      Object Details
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="px-3 pb-3 pt-3 grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <div className="text-xs text-slate-500 mb-0.5">ID</div>
                      <div className="font-mono text-xs text-slate-700">{currentObject.id}</div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 mb-0.5">Source</div>
                      <div className="flex items-center">
                        <span className="text-xs text-slate-700">{currentObject.detection_source}</span>
                        {currentObject.detection_source === 'ML Detection' && (
                          <span className="ml-1 px-1 py-0.5 bg-blue-50 text-blue-700 rounded text-[10px]">
                            {Math.round(currentObject.confidence * 100)}%
                          </span>
                        )}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 mb-0.5">Name / Species</div>
                      <div className="flex items-center">
                        <span className="text-xs text-slate-700">
                          {currentObject.tree_species ? `${currentObject.tree_species}` : 'Unknown'}
                        </span>
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 mb-0.5">Address</div>
                      <div className="flex items-center">
                        <span className="text-xs text-slate-700">
                          {currentObject.address ? currentObject.address : '123 Example St'}
                        </span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                <Card className="border-0 rounded-md overflow-hidden shadow-[0_1px_3px_0_rgba(6,78,59,0.1)] mt-3">
                  <CardHeader className="pb-1.5 pt-2 px-3 bg-emerald-900 rounded-none">
                    <CardTitle className="text-xs text-white uppercase tracking-wider flex items-center">
                      <Leaf className="h-3.5 w-3.5 mr-1.5" />
                      Species Identification
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="px-3 py-4">
                    <div className="text-center">
                      <p className="text-sm text-emerald-800 font-semibold mb-3">
                        {currentObject.tree_species || 'Unknown Species'}
                      </p>
                      <p className="text-xs text-slate-600 mb-4">
                        Species identification is based on visual recognition and database matching.
                      </p>
                      
                      <div className="grid grid-cols-2 gap-3 mt-4">
                        <Button 
                          className="flex-1 bg-emerald-800 hover:bg-emerald-700 text-white shadow-[0_1px_2px_0_rgba(6,78,59,0.1)] text-sm py-1.5 h-auto rounded-md"
                        >
                          <Search className="h-3.5 w-3.5 mr-1.5" />
                          Analyze Image
                        </Button>
                        <Button 
                          className="flex-1 bg-white hover:bg-emerald-50 text-emerald-800 text-sm px-2 py-1.5 h-auto rounded-md border border-emerald-300"
                        >
                          <FileText className="h-3.5 w-3.5 mr-1.5" />
                          View Species Data
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        )}
      </div>
      </div>
    </div>
  );
};

export default AnalyticsPanel;