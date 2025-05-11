// src/components/analytics/GeminiTreeAnalysis.jsx

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '../ui/card';
import { Button } from '../ui/button';
import { Label } from '../ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Tree, Brain, Sparkles, RefreshCw, AlertTriangle, ArrowLeft, Map } from 'lucide-react';
import { useTreeAnalysis } from '../../hooks/useTreeAnalysis';
import { TreeService } from '../../services/api/apiService';

const GeminiTreeAnalysis = ({ onBack }) => {
  const [selectedTreeId, setSelectedTreeId] = useState('');
  const [treeOptions, setTreeOptions] = useState([]);
  const [isGeminiEnabled, setIsGeminiEnabled] = useState(false);
  const [isLoadingTrees, setIsLoadingTrees] = useState(false);
  
  // Use the custom hook for tree analysis
  const { 
    treeData, 
    isLoading: isLoadingTreeData, 
    analyzeTree, 
    analysisResult, 
    isAnalyzing, 
    error: analysisError 
  } = useTreeAnalysis(selectedTreeId);
  
  // Load tree options and check if Gemini is enabled
  useEffect(() => {
    // Check if Gemini is enabled from settings
    try {
      const settings = JSON.parse(localStorage.getItem('treeRiskDashboardSettings') || '{}');
      setIsGeminiEnabled(settings.useGeminiForTreeDetection === true);
    } catch (e) {
      console.error('Error loading settings:', e);
    }
    
    // Load tree data from API
    const loadTrees = async () => {
      setIsLoadingTrees(true);
      try {
        // Try to fetch tree data from API
        const trees = await TreeService.getAllTrees();
        if (trees && trees.length > 0) {
          setTreeOptions(trees.map(tree => ({
            id: tree.id,
            name: `${tree.species || 'Unknown'} Tree (${tree.id})`
          })));
        } else {
          console.warn('No trees returned from API');
          setTreeOptions([]);
        }
      } catch (error) {
        console.error('Error loading trees:', error);
        setTreeOptions([]);
        // Show error in UI if needed
      } finally {
        setIsLoadingTrees(false);
      }
    };
    
    loadTrees();
  }, []);
  
  const handleAnalyzeTree = async () => {
    if (!selectedTreeId) return;
    try {
      await analyzeTree();
    } catch (error) {
      // Error handling is done inside the hook
      console.error('Analysis failed:', error);
    }
  };
  
  const handleSelectTree = (value) => {
    setSelectedTreeId(value);
  };
  
  const handleBack = () => {
    if (onBack) onBack();
  };
  
  const isLoading = isLoadingTrees || isLoadingTreeData || isAnalyzing;
  const error = analysisError;
  
  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex items-center mb-6">
        <Button variant="ghost" onClick={handleBack} className="mr-2 p-2">
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <h1 className="text-2xl font-bold flex items-center">
          <Brain className="h-6 w-6 mr-2 text-purple-500" />
          Gemini AI Tree Analysis
        </h1>
      </div>
      
      {!isGeminiEnabled && (
        <Card className="mb-6 border-amber-200 bg-amber-50">
          <CardContent className="pt-6">
            <div className="flex items-start">
              <AlertTriangle className="h-5 w-5 mr-2 text-amber-500 mt-0.5" />
              <div>
                <p className="font-medium text-amber-700">Gemini AI is not enabled</p>
                <p className="text-sm text-amber-600 mt-1">
                  Enable Gemini AI in Settings to use this feature. Go to Settings and check "Use Gemini AI for tree detection".
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
      
      <Card className="mb-6">
        <CardHeader className="pb-3">
          <CardTitle>Select a Tree to Analyze</CardTitle>
          <CardDescription>
            Choose a tree to get a comprehensive risk analysis powered by Gemini AI
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-4">
            <div className="space-y-2">
              <Label htmlFor="treeSelect">Select Tree</Label>
              <Select onValueChange={handleSelectTree} value={selectedTreeId}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a tree" />
                </SelectTrigger>
                <SelectContent>
                  {treeOptions.map(tree => (
                    <SelectItem key={tree.id} value={tree.id}>
                      {tree.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
        <CardFooter className="flex flex-col space-y-2">
          <Button 
            onClick={handleAnalyzeTree} 
            disabled={isLoading || !selectedTreeId || !isGeminiEnabled}
            className="w-full"
          >
            {isAnalyzing ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4 mr-2" />
                Analyze with Gemini AI
              </>
            )}
          </Button>
          
          <Button
            variant="outline"
            disabled={isLoading || !selectedTreeId || !isGeminiEnabled}
            className="w-full"
            onClick={() => {
              // This would open the map view with this tree highlighted
              if (onBack) onBack('map', selectedTreeId);
            }}
          >
            <Map className="h-4 w-4 mr-2" />
            View on Map
          </Button>
        </CardFooter>
      </Card>
      
      {error && (
        <Card className="mb-6 border-red-200 bg-red-50">
          <CardContent className="pt-6">
            <div className="flex items-start">
              <AlertTriangle className="h-5 w-5 mr-2 text-red-500 mt-0.5" />
              <div>
                <p className="font-medium text-red-700">Analysis Error</p>
                <p className="text-sm text-red-600 mt-1">{error}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
      
      {treeData && !analysisResult && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Tree className="h-5 w-5 mr-2 text-green-600" />
              Tree Information
            </CardTitle>
            <CardDescription>
              Basic information about the selected tree
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-medium text-gray-500">ID</p>
                <p>{treeData.id}</p>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-500">Species</p>
                <p>{treeData.species || 'Unknown'}</p>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-500">Height</p>
                <p>{treeData.height || 'Unknown'} m</p>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-500">Canopy Width</p>
                <p>{treeData.canopy_width || 'Unknown'} m</p>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-500">Risk Level</p>
                <p className={`font-medium ${
                  treeData.risk_level === 'high' ? 'text-red-600' : 
                  treeData.risk_level === 'medium' ? 'text-amber-600' : 
                  'text-green-600'
                }`}>
                  {treeData.risk_level ? treeData.risk_level.charAt(0).toUpperCase() + treeData.risk_level.slice(1) : 'Unknown'}
                </p>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-500">Distance to Structure</p>
                <p>{treeData.distance_to_structure || 'Unknown'} m</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
      
      {analysisResult && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Tree className="h-5 w-5 mr-2 text-green-600" />
              AI-Powered Tree Analysis Results
            </CardTitle>
            <CardDescription>
              Comprehensive risk assessment generated by Gemini AI
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="analysis">
              <TabsList className="mb-4">
                <TabsTrigger value="analysis">Risk Analysis</TabsTrigger>
                <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
                <TabsTrigger value="future">Future Concerns</TabsTrigger>
                <TabsTrigger value="comparison">Comparison</TabsTrigger>
              </TabsList>
              
              <TabsContent value="analysis" className="space-y-4">
                <div className="bg-blue-50 p-4 rounded-md">
                  <h3 className="font-medium text-blue-700 mb-2">Risk Analysis</h3>
                  <div className="text-blue-800 whitespace-pre-line">
                    {analysisResult.structured_analysis?.risk_analysis || 'No analysis available'}
                  </div>
                </div>
              </TabsContent>
              
              <TabsContent value="recommendations" className="space-y-4">
                <div className="bg-green-50 p-4 rounded-md">
                  <h3 className="font-medium text-green-700 mb-2">Recommendations</h3>
                  <div className="text-green-800 whitespace-pre-line">
                    {analysisResult.structured_analysis?.recommendations || 'No recommendations available'}
                  </div>
                </div>
              </TabsContent>
              
              <TabsContent value="future" className="space-y-4">
                <div className="bg-amber-50 p-4 rounded-md">
                  <h3 className="font-medium text-amber-700 mb-2">Future Concerns</h3>
                  <div className="text-amber-800 whitespace-pre-line">
                    {analysisResult.structured_analysis?.future_concerns || 'No future concerns identified'}
                  </div>
                </div>
              </TabsContent>
              
              <TabsContent value="comparison" className="space-y-4">
                <div className="bg-purple-50 p-4 rounded-md">
                  <h3 className="font-medium text-purple-700 mb-2">Comparison with Similar Trees</h3>
                  <div className="text-purple-800 whitespace-pre-line">
                    {analysisResult.structured_analysis?.comparison || 'No comparison data available'}
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
          <CardFooter className="flex justify-between">
            <p className="text-xs text-gray-500">
              Analysis generated by Gemini AI at {new Date(analysisResult.timestamp).toLocaleString()}
            </p>
            <Button variant="outline" size="sm" onClick={() => analyzeTree()}>
              Refresh Analysis
            </Button>
          </CardFooter>
        </Card>
      )}
    </div>
  );
};

export default GeminiTreeAnalysis;