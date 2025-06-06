// src/components/analytics/AnalyticsDashboard.jsx

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { 
  AlertTriangle, 
  BarChart, 
  TrendingUp, 
  Calendar, 
  Tree, 
  Map, 
  Filter,
  Download
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  BarChart as RechartsBarChart,
  Bar,
  PieChart, 
  Pie, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  Cell
} from 'recharts';

const AnalyticsDashboard = () => {
  const [timeframe, setTimeframe] = useState('month');
  const [riskFilter, setRiskFilter] = useState('all');
  const [isLoading, setIsLoading] = useState(true);
  const [analyticsData, setAnalyticsData] = useState(null);

  useEffect(() => {
    // Mock data loading
    const loadData = async () => {
      setIsLoading(true);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Generate mock data
      const mockData = generateMockData(timeframe, riskFilter);
      setAnalyticsData(mockData);
      
      setIsLoading(false);
    };
    
    loadData();
  }, [timeframe, riskFilter]);

  // Function to generate mock analytics data
  const generateMockData = (timeframe, riskFilter) => {
    // Risk distribution data
    const riskDistribution = [
      { name: 'High Risk', value: 24, color: '#ef4444' },
      { name: 'Medium Risk', value: 45, color: '#f97316' },
      { name: 'Low Risk', value: 87, color: '#22c55e' }
    ];
    
    // Time series data based on selected timeframe
    let timeSeries = [];
    if (timeframe === 'week') {
      timeSeries = [
        { name: 'Mon', assessments: 5, highRisk: 1 },
        { name: 'Tue', assessments: 7, highRisk: 2 },
        { name: 'Wed', assessments: 10, highRisk: 3 },
        { name: 'Thu', assessments: 8, highRisk: 2 },
        { name: 'Fri', assessments: 12, highRisk: 4 },
        { name: 'Sat', assessments: 3, highRisk: 1 },
        { name: 'Sun', assessments: 2, highRisk: 0 }
      ];
    } else if (timeframe === 'month') {
      timeSeries = [
        { name: 'Week 1', assessments: 28, highRisk: 7 },
        { name: 'Week 2', assessments: 32, highRisk: 8 },
        { name: 'Week 3', assessments: 22, highRisk: 5 },
        { name: 'Week 4', assessments: 35, highRisk: 10 }
      ];
    } else {
      timeSeries = [
        { name: 'Jan', assessments: 105, highRisk: 22 },
        { name: 'Feb', assessments: 120, highRisk: 25 },
        { name: 'Mar', assessments: 150, highRisk: 30 },
        { name: 'Apr', assessments: 135, highRisk: 28 },
        { name: 'May', assessments: 110, highRisk: 20 },
        { name: 'Jun', assessments: 95, highRisk: 18 },
        { name: 'Jul', assessments: 105, highRisk: 22 },
        { name: 'Aug', assessments: 120, highRisk: 25 },
        { name: 'Sep', assessments: 130, highRisk: 27 },
        { name: 'Oct', assessments: 140, highRisk: 29 },
        { name: 'Nov', assessments: 125, highRisk: 26 },
        { name: 'Dec', assessments: 115, highRisk: 24 }
      ];
    }
    
    // Property risk data
    const propertyRiskData = [
      { name: '123 Oak St', highRisk: 4, mediumRisk: 7, lowRisk: 12 },
      { name: '456 Elm St', highRisk: 2, mediumRisk: 5, lowRisk: 8 },
      { name: '789 Pine St', highRisk: 5, mediumRisk: 3, lowRisk: 6 },
      { name: '321 Maple Dr', highRisk: 1, mediumRisk: 9, lowRisk: 14 },
      { name: '654 Cedar Ln', highRisk: 3, mediumRisk: 4, lowRisk: 9 }
    ];
    
    // Species data
    const speciesData = [
      { name: 'Oak', count: 45, highRisk: 12 },
      { name: 'Pine', count: 32, highRisk: 8 },
      { name: 'Maple', count: 28, highRisk: 7 },
      { name: 'Cedar', count: 19, highRisk: 4 },
      { name: 'Elm', count: 14, highRisk: 3 },
      { name: 'Other', count: 18, highRisk: 5 }
    ];
    
    return {
      riskDistribution,
      timeSeries,
      propertyRiskData,
      speciesData,
      summary: {
        totalTrees: 156,
        highRiskTrees: 24,
        pendingValidations: 18,
        recentAssessments: 35
      }
    };
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading analytics data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Tree Risk Assessment Analytics</h1>
        
        <div className="flex space-x-4">
          {/* Timeframe selector */}
          <div className="flex items-center space-x-2">
            <Calendar className="h-5 w-5 text-gray-500" />
            <select 
              className="border rounded-md p-2"
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
            >
              <option value="week">Past Week</option>
              <option value="month">Past Month</option>
              <option value="year">Past Year</option>
            </select>
          </div>
          
          {/* Risk filter */}
          <div className="flex items-center space-x-2">
            <Filter className="h-5 w-5 text-gray-500" />
            <select 
              className="border rounded-md p-2"
              value={riskFilter}
              onChange={(e) => setRiskFilter(e.target.value)}
            >
              <option value="all">Low Risk</option>
              <option value="high">High Risk Only</option>
              <option value="medium">Medium Risk & Above</option>
            </select>
          </div>
          
          {/* Export button */}
          <button className="flex items-center space-x-2 px-3 py-2 bg-green-100 text-green-700 rounded-md hover:bg-green-200">
            <Download className="h-5 w-5" />
            <span>Export Data</span>
          </button>
        </div>
      </div>
      
      {/* Summary cards */}
      <div className="grid grid-cols-4 gap-6">
        <Card>
          <CardContent className="pt-6">
            <div className="flex justify-between items-center">
              <div>
                <p className="text-gray-500">Total Trees</p>
                <p className="text-3xl font-bold">{analyticsData.summary.totalTrees}</p>
              </div>
              <div className="h-12 w-12 bg-blue-100 rounded-full flex items-center justify-center">
                <Tree className="h-6 w-6 text-blue-600" />
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-6">
            <div className="flex justify-between items-center">
              <div>
                <p className="text-gray-500">High Risk Trees</p>
                <p className="text-3xl font-bold">{analyticsData.summary.highRiskTrees}</p>
              </div>
              <div className="h-12 w-12 bg-red-100 rounded-full flex items-center justify-center">
                <AlertTriangle className="h-6 w-6 text-red-600" />
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-6">
            <div className="flex justify-between items-center">
              <div>
                <p className="text-gray-500">Pending Validations</p>
                <p className="text-3xl font-bold">{analyticsData.summary.pendingValidations}</p>
              </div>
              <div className="h-12 w-12 bg-yellow-100 rounded-full flex items-center justify-center">
                <TrendingUp className="h-6 w-6 text-yellow-600" />
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-6">
            <div className="flex justify-between items-center">
              <div>
                <p className="text-gray-500">Recent Assessments</p>
                <p className="text-3xl font-bold">{analyticsData.summary.recentAssessments}</p>
              </div>
              <div className="h-12 w-12 bg-purple-100 rounded-full flex items-center justify-center">
                <BarChart className="h-6 w-6 text-purple-600" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* Charts */}
      <div className="grid grid-cols-2 gap-6">
        {/* Risk Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Risk Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={analyticsData.riskDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {analyticsData.riskDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
        
        {/* Assessments Over Time */}
        <Card>
          <CardHeader>
            <CardTitle>Assessments Over Time</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={analyticsData.timeSeries}
                  margin={{
                    top: 5,
                    right: 30,
                    left: 20,
                    bottom: 5,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="assessments"
                    stroke="#3b82f6"
                    activeDot={{ r: 8 }}
                    name="Total Assessments"
                  />
                  <Line type="monotone" dataKey="highRisk" stroke="#ef4444" name="High Risk Identified" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
        
        {/* Property Risk Breakdown */}
        <Card>
          <CardHeader>
            <CardTitle>Property Risk Breakdown</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsBarChart
                  layout="vertical"
                  data={analyticsData.propertyRiskData}
                  margin={{
                    top: 5,
                    right: 30,
                    left: 40,
                    bottom: 5,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="name" type="category" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="highRisk" stackId="a" fill="#ef4444" name="High Risk" />
                  <Bar dataKey="mediumRisk" stackId="a" fill="#f97316" name="Medium Risk" />
                  <Bar dataKey="lowRisk" stackId="a" fill="#22c55e" name="Low Risk" />
                </RechartsBarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
        
        {/* Species Data */}
        <Card>
          <CardHeader>
            <CardTitle>Tree Species Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsBarChart
                  data={analyticsData.speciesData}
                  margin={{
                    top: 5,
                    right: 30,
                    left: 20,
                    bottom: 5,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="count" fill="#3b82f6" name="Total Count" />
                  <Bar dataKey="highRisk" fill="#ef4444" name="High Risk" />
                </RechartsBarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default AnalyticsDashboard;