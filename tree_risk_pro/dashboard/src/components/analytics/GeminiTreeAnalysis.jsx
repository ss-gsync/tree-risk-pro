// src/components/analytics/GeminiTreeAnalysis.jsx

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '../ui/card';
import { Button } from '../ui/button';
import { Label } from '../ui/label';
import { Input } from '../ui/input';
import { Textarea } from '../ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Tree, Brain, Sparkles, RefreshCw, AlertTriangle, ArrowLeft } from 'lucide-react';
import { analyzeTreeWithGemini } from '../../services/api/geminiService';

const GeminiTreeAnalysis = ({ onBack }) => {
  const [selectedTreeId, setSelectedTreeId] = useState('');
  const [treeOptions, setTreeOptions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isGeminiEnabled, setIsGeminiEnabled] = useState(false);
  
  // Load tree options from localStorage or mock data
  useEffect(() => {
    // Check if Gemini is enabled
    try {
      const settings = JSON.parse(localStorage.getItem('treeRiskDashboardSettings') || '{}');
      setIsGeminiEnabled(settings.useGeminiForTreeDetection === true);
    } catch (e) {
      console.error('Error loading settings:', e);
    }
    
    // Load tree options
    const mockTreeOptions = [
      { id: 'tree_001', name: 'Oak Tree (123 Main St)' },
      { id: 'tree_002', name: 'Pine Tree (456 Elm Ave)' },
      { id: 'tree_003', name: 'Maple Tree (789 Oak Dr)' },
      { id: 'tree_004', name: 'Cedar Tree (321 Willow Ln)' },
      { id: 'tree_005', name: 'Birch Tree (654 Maple St)' },
    ];
    setTreeOptions(mockTreeOptions);
  }, []);
  
  const handleAnalyzeTree = async () => {
    if (!selectedTreeId) {
      setError('Please select a tree to analyze');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      // Find the selected tree in options
      const selectedTree = treeOptions.find(t => t.id === selectedTreeId);
      
      // Mock tree data for analysis
      const treeData = {
        id: selectedTree.id,
        species: selectedTree.name.split(' ')[0], // Extract species from name
        height: Math.floor(Math.random() * 30) + 10, // Random height between 10-40m
        canopy_width: Math.floor(Math.random() * 10) + 5, // Random width between 5-15m
        risk_level: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)],
        risk_factors: [
          { factor: 'Proximity to structure', level: 'medium', distance: '8m' },
          { factor: 'Deadwood', level: 'low', description: 'Minor deadwood in crown' },
          { factor: 'Root health', level: 'medium', description: 'Some root compression' }
        ],
        assessment_history: [
          { date: '2023-05-15', condition: 'good', notes: 'Annual inspection' },
          { date: '2023-11-10', condition: 'fair', notes: 'Storm damage assessment' }
        ],
        distance_to_structure: Math.floor(Math.random() * 15) + 3, // Random distance between 3-18m
        property_type: ['Residential', 'Commercial', 'Public Park'][Math.floor(Math.random() * 3)]
      };
      
      // Call Gemini API for analysis
      const result = await analyzeTreeWithGemini(treeData);
      
      if (result.success) {
        setAnalysisResult(result);
      } else {
        setError(result.message || 'Failed to analyze tree');
      }
    } catch (err) {
      setError(`Analysis error: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleSelectTree = (value) => {
    setSelectedTreeId(value);
    setAnalysisResult(null); // Clear previous results
    setError(null);
  };
  
  const handleBack = () => {
    if (onBack) onBack();
  };
  
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
        <CardFooter>
          <Button 
            onClick={handleAnalyzeTree} 
            disabled={isLoading || !selectedTreeId || !isGeminiEnabled}
            className="w-full"
          >
            {isLoading ? (
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
      
      {analysisResult && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Tree className="h-5 w-5 mr-2 text-green-600" />
              Tree Analysis Results
            </CardTitle>
            <CardDescription>
              AI-powered risk assessment and recommendations
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
                    {analysisResult.structured_analysis.risk_analysis || 'No analysis available'}
                  </div>
                </div>
              </TabsContent>
              
              <TabsContent value="recommendations" className="space-y-4">
                <div className="bg-green-50 p-4 rounded-md">
                  <h3 className="font-medium text-green-700 mb-2">Recommendations</h3>
                  <div className="text-green-800 whitespace-pre-line">
                    {analysisResult.structured_analysis.recommendations || 'No recommendations available'}
                  </div>
                </div>
              </TabsContent>
              
              <TabsContent value="future" className="space-y-4">
                <div className="bg-amber-50 p-4 rounded-md">
                  <h3 className="font-medium text-amber-700 mb-2">Future Concerns</h3>
                  <div className="text-amber-800 whitespace-pre-line">
                    {analysisResult.structured_analysis.future_concerns || 'No future concerns identified'}
                  </div>
                </div>
              </TabsContent>
              
              <TabsContent value="comparison" className="space-y-4">
                <div className="bg-purple-50 p-4 rounded-md">
                  <h3 className="font-medium text-purple-700 mb-2">Comparison with Similar Trees</h3>
                  <div className="text-purple-800 whitespace-pre-line">
                    {analysisResult.structured_analysis.comparison || 'No comparison data available'}
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
          <CardFooter className="flex justify-between">
            <p className="text-xs text-gray-500">
              Analysis generated by Gemini AI at {new Date(analysisResult.timestamp).toLocaleString()}
            </p>
            <Button variant="outline" size="sm" onClick={() => setAnalysisResult(null)}>
              Clear Results
            </Button>
          </CardFooter>
        </Card>
      )}
    </div>
  );
};

export default GeminiTreeAnalysis;