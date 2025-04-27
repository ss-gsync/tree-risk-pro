/**
 * Development Server for Tree Risk Dashboard
 * 
 * This Express server provides mock data for frontend development.
 * It serves JSON files from the mock data directory and handles basic API routes.
 */

const express = require('express');
const path = require('path');
const cors = require('cors');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 5000;

// Enable CORS for all routes
app.use(cors());

// Parse JSON bodies
app.use(express.json());

// Serve static files from the data directory
app.use('/data', express.static(path.join(__dirname, 'mock_data')));

// API Routes

// Properties
app.get('/api/properties', (req, res) => {
  try {
    const data = fs.readFileSync(path.join(__dirname, 'mock_data/properties.json'), 'utf8');
    res.json(JSON.parse(data));
  } catch (err) {
    console.error('Error reading properties data:', err);
    res.status(500).json({ error: 'Failed to load properties' });
  }
});

app.get('/api/properties/:id', (req, res) => {
  try {
    const data = fs.readFileSync(path.join(__dirname, 'mock_data/properties.json'), 'utf8');
    const properties = JSON.parse(data);
    const property = properties.find(p => p.id === req.params.id);
    
    if (!property) {
      return res.status(404).json({ error: 'Property not found' });
    }
    
    res.json(property);
  } catch (err) {
    console.error('Error reading property data:', err);
    res.status(500).json({ error: 'Failed to load property' });
  }
});

// Trees
app.get('/api/trees', (req, res) => {
  try {
    const data = fs.readFileSync(path.join(__dirname, 'mock_data/trees.json'), 'utf8');
    res.json(JSON.parse(data));
  } catch (err) {
    console.error('Error reading trees data:', err);
    res.status(500).json({ error: 'Failed to load trees' });
  }
});

app.get('/api/trees/:id', (req, res) => {
  try {
    const data = fs.readFileSync(path.join(__dirname, 'mock_data/trees.json'), 'utf8');
    const trees = JSON.parse(data);
    const tree = trees.find(t => t.id === req.params.id);
    
    if (!tree) {
      return res.status(404).json({ error: 'Tree not found' });
    }
    
    res.json(tree);
  } catch (err) {
    console.error('Error reading tree data:', err);
    res.status(500).json({ error: 'Failed to load tree' });
  }
});

app.get('/api/properties/:id/trees', (req, res) => {
  try {
    const data = fs.readFileSync(path.join(__dirname, 'mock_data/trees.json'), 'utf8');
    const trees = JSON.parse(data);
    const propertyTrees = trees.filter(t => t.property_id === req.params.id);
    
    res.json(propertyTrees);
  } catch (err) {
    console.error('Error reading property trees data:', err);
    res.status(500).json({ error: 'Failed to load property trees' });
  }
});

// Validation Queue
app.get('/api/validation/queue', (req, res) => {
  try {
    const data = fs.readFileSync(path.join(__dirname, 'mock_data/validation_queue.json'), 'utf8');
    const validationItems = JSON.parse(data);
    
    // Apply filters if provided
    let filtered = validationItems;
    if (req.query.status) {
      filtered = filtered.filter(item => item.status === req.query.status);
    }
    if (req.query.property_id) {
      filtered = filtered.filter(item => item.property_id === req.query.property_id);
    }
    
    res.json(filtered);
  } catch (err) {
    console.error('Error reading validation queue data:', err);
    res.status(500).json({ error: 'Failed to load validation queue' });
  }
});

// LiDAR Data
app.get('/api/lidar/property/:id', (req, res) => {
  try {
    // Load property data
    const propertiesData = fs.readFileSync(path.join(__dirname, 'mock_data/properties.json'), 'utf8');
    const properties = JSON.parse(propertiesData);
    const property = properties.find(p => p.id === req.params.id);
    
    if (!property) {
      return res.status(404).json({ error: 'Property not found' });
    }
    
    // Load trees data
    const treesData = fs.readFileSync(path.join(__dirname, 'mock_data/trees.json'), 'utf8');
    const trees = JSON.parse(treesData);
    const propertyTrees = trees.filter(t => t.property_id === req.params.id);
    
    // Create mock LiDAR response
    const lidarResponse = {
      processed_at: new Date().toISOString(),
      points_processed: 1000000 + Math.round(Math.random() * 500000),
      trees_detected: propertyTrees.length,
      trees: propertyTrees.map((tree, i) => ({
        id: tree.id,
        position: (tree.location || property.center).concat([0]), // Add Z coordinate
        height: tree.height,
        canopy_width: tree.canopy_width || 10,
        features: {
          trunk_diameter: (tree.dbh || 24) / 12, // Convert to feet
          branches: 5 + (i % 3)
        }
      }))
    };
    
    res.json(lidarResponse);
  } catch (err) {
    console.error('Error creating LiDAR data:', err);
    res.status(500).json({ error: 'Failed to generate LiDAR data' });
  }
});

/**
 * Configuration endpoint
 * Provides application settings to the frontend
 */
app.get('/api/config', (req, res) => {
  res.json({
    mode: 'test',
    version: '1.0.0',
    useTestData: true
  });
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
  console.log(`Mock data available at http://localhost:${PORT}/data`);
});