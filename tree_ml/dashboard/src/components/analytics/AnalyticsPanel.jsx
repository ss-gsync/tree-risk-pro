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
  X
} from 'lucide-react';
import { useGeminiTreeDetection } from '../../hooks/useGeminiTreeDetection';

/**
 * AnalyticsPanel - Side panel for object risk analysis
 * This component displays detailed risk analysis for detected objects
 * and integrates with the ValidationQueue and Detection systems
 */
const AnalyticsPanel = ({
  treeData = null,
  selectedObject = null,
  onSaveToQueue = () => {},
  className = ""
}) => {
  const [activeTab, setActiveTab] = useState('overview');
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
  
  return (
    <div className="p-0 bg-white overflow-auto h-full flex flex-col relative">
      {/* Navigation tabs */}
      <div className="bg-gray-50 px-4 pt-2 pb-0">
        <div className="flex space-x-1 border-b border-gray-200">
          <button 
            className={`px-3 py-1.5 text-xs font-medium ${
              activeTab === 'overview' 
                ? 'border-b-2 border-gray-900 text-gray-900 bg-gray-50' 
                : 'text-gray-700 hover:text-gray-900 hover:bg-gray-50'
            }`}
            onClick={() => setActiveTab('overview')}
          >
            Object Details
          </button>
          <button 
            className={`px-3 py-1.5 text-xs font-medium ${
              activeTab === 'analysis' 
                ? 'border-b-2 border-gray-900 text-gray-900 bg-gray-50' 
                : 'text-gray-700 hover:text-gray-900 hover:bg-gray-50'
            }`}
            onClick={() => setActiveTab('analysis')}
          >
            Risk Analysis
          </button>
        </div>
      </div>
      
      {/* Content area */}
      <div className="flex-1 overflow-auto pb-16">
        {activeTab === 'overview' && (
          <div className="space-y-3 px-4 py-3">
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
                <Card className="mb-3">
                  <CardHeader className="pb-2 pt-3 px-3 bg-gray-50">
                    <CardTitle className="text-xs text-gray-900 uppercase tracking-wider flex items-center justify-between">
                      <span>Objects for Validation ({detectedObjects.length})</span>
                      <button 
                        className="text-xs text-purple-900 hover:text-purple-800 flex items-center"
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
                                ? 'border-purple-900 bg-gray-50' 
                                : 'border-slate-200 hover:border-purple-300 hover:bg-gray-50/30'
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
                <Card className="border border-gray-200 rounded-none overflow-hidden">
                  <CardHeader className="pb-1.5 pt-2 px-3 bg-purple-950 rounded-none">
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
                <Card>
                  <CardHeader className="pb-2 pt-3 px-3 bg-gray-50">
                    <CardTitle className="text-xs text-gray-900 uppercase tracking-wider flex items-center justify-between">
                      <span>Risk Analysis</span>
                      {currentObject.analysis_status === 'complete' ? (
                        <span className="bg-green-50 text-green-700 px-1.5 py-0.5 rounded text-[10px] font-medium">Complete</span>
                      ) : (
                        <span className="bg-amber-50 text-amber-700 px-1.5 py-0.5 rounded text-[10px] font-medium">Needs Analysis</span>
                      )}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="px-3 pb-3">
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
                            className="flex-1 bg-purple-900 hover:bg-purple-800 text-white shadow-sm text-sm py-1.5 h-auto"
                            onClick={() => setActiveTab('analysis')}
                          >
                            <CircleAlert className="h-3.5 w-3.5 mr-1.5" />
                            Edit Analysis
                          </Button>
                          <Button 
                            className="flex-1 bg-white hover:bg-gray-100 text-neutral-800 text-sm px-2 py-1.5 h-auto border border-neutral-400"
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
                        
                        <div className="absolute bottom-0 left-0 right-0 bg-white p-3 border-t border-gray-200 grid grid-cols-2 gap-2">
                          <Button 
                            className="flex-1 bg-purple-950 hover:bg-purple-900 text-white shadow-sm text-sm py-2 h-auto rounded border border-gray-700"
                            onClick={() => setActiveTab('analysis')}
                          >
                            <CircleAlert className="h-3.5 w-3.5 mr-1.5" />
                            Start Analysis
                          </Button>
                          
                          <Button 
                            className="flex-1 bg-gray-100 hover:bg-gray-200 text-purple-950 font-semibold text-sm py-2 h-auto rounded border border-gray-700"
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
                
                
                {/* Gemini Analysis Card - shown if analysis has been run */}
                {currentObject?.gemini_analysis && (
                  <Card>
                    <CardHeader className="pb-2 pt-3 px-3 bg-gray-50">
                      <CardTitle className="text-xs text-gray-900 uppercase tracking-wider flex items-center">
                        <Brain className="h-3.5 w-3.5 mr-1.5 text-purple-900" />
                        Gemini AI Analysis
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="px-3 pb-3">
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
                
              </>
            )}
          </div>
        )}
        
        {activeTab === 'analysis' && (
          <div className="space-y-3 px-4 py-3">
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
                <Card>
                  <CardHeader className="pb-2 pt-3 px-3 bg-gray-50">
                    <CardTitle className="text-xs text-gray-900 uppercase tracking-wider">
                      Risk Assessment
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="px-3 pb-3">
                    <div className="space-y-4">
                      <div className="space-y-1.5">
                        <label className="text-xs text-slate-700 font-medium">Object Status</label>
                        <div className={`flex items-center rounded-md p-1.5 ${
                          riskLevel === 'high' ? 'bg-red-50 border border-red-200' :
                          riskLevel === 'medium' ? 'bg-orange-50 border border-orange-200' :
                          riskLevel === 'low' ? 'bg-yellow-100 border border-yellow-200' :
                          'bg-green-50 border border-green-200'
                        }`}>
                          <div className="flex-1 flex items-center">
                            <span className={`w-2 h-2 rounded-full mr-2 ${
                              riskLevel === 'high' ? 'bg-red-500' :
                              riskLevel === 'medium' ? 'bg-orange-500' :
                              riskLevel === 'low' ? 'bg-yellow-500' :
                              'bg-green-500'
                            }`}></span>
                            <span className="text-xs font-medium">
                              {riskLevel === 'high' ? 'High Risk' :
                                riskLevel === 'medium' ? 'Medium Risk' :
                                riskLevel === 'low' ? 'Low Risk' :
                                'No Risk'}
                            </span>
                          </div>
                          <div className="text-[10px] text-slate-600">
                            {getSelectedFactorCount()} factor{getSelectedFactorCount() !== 1 ? 's' : ''} selected
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-2">
                        <label className="text-xs text-slate-700 font-medium">Risk Factors</label>
                        <div className="space-y-1.5">
                          <div 
                            className={`flex items-center justify-between p-1.5 rounded-md cursor-pointer ${
                              selectedRiskFactors.treeHealth ? 'bg-red-50 border border-red-200' : 'bg-slate-50 border border-slate-200 hover:bg-slate-100'
                            }`}
                            onClick={() => handleRiskFactorChange('treeHealth')}
                          >
                            <div className="flex items-center">
                              <AlertTriangle className="h-3.5 w-3.5 mr-1.5 text-slate-700" />
                              <span className="text-xs">Tree Health Issues</span>
                              <span className="ml-1 text-[9px] text-slate-500">(dead/diseased)</span>
                            </div>
                            <div className={`w-4 h-4 rounded-full flex items-center justify-center ${
                              selectedRiskFactors.treeHealth ? 'bg-red-500 text-white' : 'border border-slate-300'
                            }`}>
                              {selectedRiskFactors.treeHealth && (
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="4">
                                  <polyline points="20 6 9 17 4 12"></polyline>
                                </svg>
                              )}
                            </div>
                          </div>
                          
                          <div 
                            className={`flex items-center justify-between p-1.5 rounded-md cursor-pointer ${
                              selectedRiskFactors.structuralDefects ? 'bg-red-50 border border-red-200' : 'bg-slate-50 border border-slate-200 hover:bg-slate-100'
                            }`}
                            onClick={() => handleRiskFactorChange('structuralDefects')}
                          >
                            <div className="flex items-center">
                              <AlertTriangle className="h-3.5 w-3.5 mr-1.5 text-slate-700" />
                              <span className="text-xs">Structural Defects</span>
                              <span className="ml-1 text-[9px] text-slate-500">(cracks/breaks)</span>
                            </div>
                            <div className={`w-4 h-4 rounded-full flex items-center justify-center ${
                              selectedRiskFactors.structuralDefects ? 'bg-red-500 text-white' : 'border border-slate-300'
                            }`}>
                              {selectedRiskFactors.structuralDefects && (
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="4">
                                  <polyline points="20 6 9 17 4 12"></polyline>
                                </svg>
                              )}
                            </div>
                          </div>
                          
                          <div 
                            className={`flex items-center justify-between p-1.5 rounded-md cursor-pointer ${
                              selectedRiskFactors.leaningAngle ? 'bg-orange-50 border border-orange-200' : 'bg-slate-50 border border-slate-200 hover:bg-slate-100'
                            }`}
                            onClick={() => handleRiskFactorChange('leaningAngle')}
                          >
                            <div className="flex items-center">
                              <Ruler className="h-3.5 w-3.5 mr-1.5 text-slate-700" />
                              <span className="text-xs">Leaning Angle</span>
                              <span className="ml-1 text-[9px] text-slate-500">(15-40°+)</span>
                            </div>
                            <div className={`w-4 h-4 rounded-full flex items-center justify-center ${
                              selectedRiskFactors.leaningAngle ? 'bg-orange-500 text-white' : 'border border-slate-300'
                            }`}>
                              {selectedRiskFactors.leaningAngle && (
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="4">
                                  <polyline points="20 6 9 17 4 12"></polyline>
                                </svg>
                              )}
                            </div>
                          </div>
                          
                          <div 
                            className={`flex items-center justify-between p-1.5 rounded-md cursor-pointer ${
                              selectedRiskFactors.treeSize ? 'bg-orange-50 border border-orange-200' : 'bg-slate-50 border border-slate-200 hover:bg-slate-100'
                            }`}
                            onClick={() => handleRiskFactorChange('treeSize')}
                          >
                            <div className="flex items-center">
                              <Ruler className="h-3.5 w-3.5 mr-1.5 text-slate-700" />
                              <span className="text-xs">Tree Size</span>
                              <span className="ml-1 text-[9px] text-slate-500">(SM/M/LG)</span>
                            </div>
                            <div className={`w-4 h-4 rounded-full flex items-center justify-center ${
                              selectedRiskFactors.treeSize ? 'bg-orange-500 text-white' : 'border border-slate-300'
                            }`}>
                              {selectedRiskFactors.treeSize && (
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="4">
                                  <polyline points="20 6 9 17 4 12"></polyline>
                                </svg>
                              )}
                            </div>
                          </div>
                          
                          <div 
                            className={`flex items-center justify-between p-1.5 rounded-md cursor-pointer ${
                              selectedRiskFactors.proximityToStructures ? 'bg-red-50 border border-red-200' : 'bg-slate-50 border border-slate-200 hover:bg-slate-100'
                            }`}
                            onClick={() => handleRiskFactorChange('proximityToStructures')}
                          >
                            <div className="flex items-center">
                              <Smartphone className="h-3.5 w-3.5 mr-1.5 text-slate-700" />
                              <span className="text-xs">Near High-Risk Areas</span>
                              <span className="ml-1 text-[9px] text-slate-500">(buildings/power)</span>
                            </div>
                            <div className={`w-4 h-4 rounded-full flex items-center justify-center ${
                              selectedRiskFactors.proximityToStructures ? 'bg-red-500 text-white' : 'border border-slate-300'
                            }`}>
                              {selectedRiskFactors.proximityToStructures && (
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="4">
                                  <polyline points="20 6 9 17 4 12"></polyline>
                                </svg>
                              )}
                            </div>
                          </div>
                          
                          <div 
                            className={`flex items-center justify-between p-1.5 rounded-md cursor-pointer ${
                              selectedRiskFactors.canopyDensity ? 'bg-orange-50 border border-orange-200' : 'bg-slate-50 border border-slate-200 hover:bg-slate-100'
                            }`}
                            onClick={() => handleRiskFactorChange('canopyDensity')}
                          >
                            <div className="flex items-center">
                              <TreePine className="h-3.5 w-3.5 mr-1.5 text-slate-700" />
                              <span className="text-xs">Canopy Density Issues</span>
                            </div>
                            <div className={`w-4 h-4 rounded-full flex items-center justify-center ${
                              selectedRiskFactors.canopyDensity ? 'bg-orange-500 text-white' : 'border border-slate-300'
                            }`}>
                              {selectedRiskFactors.canopyDensity && (
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="4">
                                  <polyline points="20 6 9 17 4 12"></polyline>
                                </svg>
                              )}
                            </div>
                          </div>
                          
                          <div 
                            className={`flex items-center justify-between p-1.5 rounded-md cursor-pointer ${
                              selectedRiskFactors.deadLimbs ? 'bg-red-50 border border-red-200' : 'bg-slate-50 border border-slate-200 hover:bg-slate-100'
                            }`}
                            onClick={() => handleRiskFactorChange('deadLimbs')}
                          >
                            <div className="flex items-center">
                              <AlertTriangle className="h-3.5 w-3.5 mr-1.5 text-slate-700" />
                              <span className="text-xs">Dead/Hazardous Limbs</span>
                            </div>
                            <div className={`w-4 h-4 rounded-full flex items-center justify-center ${
                              selectedRiskFactors.deadLimbs ? 'bg-red-500 text-white' : 'border border-slate-300'
                            }`}>
                              {selectedRiskFactors.deadLimbs && (
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="4">
                                  <polyline points="20 6 9 17 4 12"></polyline>
                                </svg>
                              )}
                            </div>
                          </div>
                          
                          <div 
                            className={`flex items-center justify-between p-1.5 rounded-md cursor-pointer ${
                              selectedRiskFactors.rootSoilIssues ? 'bg-orange-50 border border-orange-200' : 'bg-slate-50 border border-slate-200 hover:bg-slate-100'
                            }`}
                            onClick={() => handleRiskFactorChange('rootSoilIssues')}
                          >
                            <div className="flex items-center">
                              <Layers className="h-3.5 w-3.5 mr-1.5 text-slate-700" />
                              <span className="text-xs">Root/Soil Condition</span>
                            </div>
                            <div className={`w-4 h-4 rounded-full flex items-center justify-center ${
                              selectedRiskFactors.rootSoilIssues ? 'bg-orange-500 text-white' : 'border border-slate-300'
                            }`}>
                              {selectedRiskFactors.rootSoilIssues && (
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="4">
                                  <polyline points="20 6 9 17 4 12"></polyline>
                                </svg>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {isGeminiEnabled && (
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <label className="text-xs text-slate-700 font-medium">AI Analysis Prompt</label>
                            <span className="text-[10px] text-purple-900 bg-gray-50 px-1.5 py-0.5 rounded">
                              Gemini AI Enabled
                            </span>
                          </div>
                          <div className="rounded-md border border-slate-200 bg-white">
                            <Textarea 
                              className="text-xs min-h-[150px] font-mono bg-white border-0 resize-y"
                              placeholder="Select risk factors above to generate a customized prompt..."
                              value={customQuery}
                              onChange={(e) => setCustomQuery(e.target.value)}
                            />
                            <div className="p-2 border-t border-slate-200 bg-slate-50 rounded-b-md">
                              <div className="text-[10px] text-slate-600 flex items-center">
                                <Info className="h-3 w-3 mr-1 text-slate-400" />
                                Customize this prompt to focus Gemini AI on specific risk factors. The AI analysis will help fill in details for validation.
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </CardContent>
                  <CardFooter className="px-0 pb-16 pt-0">
                    <div className="absolute bottom-0 left-0 right-0 bg-white p-3 border-t border-gray-200 grid grid-cols-2 gap-2">
                      <Button 
                        className="flex-1 bg-purple-950 hover:bg-purple-900 text-white shadow-sm text-sm py-2 h-auto rounded border border-gray-700"
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
                            Start Analysis
                          </>
                        )}
                      </Button>
                      <Button 
                        className="flex-1 bg-gray-100 hover:bg-gray-200 text-purple-950 font-semibold text-sm py-2 h-auto rounded border border-gray-700"
                        onClick={() => {
                          // Display a dialog or modal to review the prompt
                          alert("Review Prompt: " + customQuery);
                        }}
                      >
                        <FileCheck className="h-3.5 w-3.5 mr-1.5" />
                        Review Prompt
                      </Button>
                    </div>
                  </CardFooter>
                </Card>
                
                <Card>
                  <CardHeader className="pb-2 pt-3 px-3 bg-gray-50">
                    <CardTitle className="text-xs text-gray-900 uppercase tracking-wider">
                      Risk Level Guidelines
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="px-3 pb-3 text-xs space-y-1.5">
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
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalyticsPanel;