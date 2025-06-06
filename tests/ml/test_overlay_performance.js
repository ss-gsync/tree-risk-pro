#!/usr/bin/env node

/**
 * Test script for measuring performance of optimized ML overlay
 * and detection scripts compared to the original versions.
 */

const fs = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');

// Use a specific detection ID for consistent testing
const TEST_DETECTION_ID = '1748726908';
const DATA_DIR = path.join(__dirname, '..', '..', 'data', 'tests');

console.log('ML Overlay Performance Test');
console.log('==========================');
console.log(`Testing with detection ID: ${TEST_DETECTION_ID}`);

async function runTest() {
  try {
    // Load detection data
    const startLoadTime = performance.now();
    const detectionPath = path.join(DATA_DIR, `test_${TEST_DETECTION_ID}`, 'ml_response', 'trees.json');
    
    if (!fs.existsSync(detectionPath)) {
      console.error(`Detection file not found: ${detectionPath}`);
      console.log('Available test files:');
      const detectionDirs = fs.readdirSync(DATA_DIR)
        .filter(name => name.startsWith('test_'));
      detectionDirs.forEach(dir => {
        const treesPath = path.join(DATA_DIR, dir, 'ml_response', 'trees.json');
        if (fs.existsSync(treesPath)) {
          console.log(`- ${dir}: ${formatBytes(fs.statSync(treesPath).size)}`);
        }
      });
      return;
    }
    
    console.log(`Loading detection data from: ${detectionPath}`);
    const detectionData = JSON.parse(fs.readFileSync(detectionPath, 'utf8'));
    const loadTime = performance.now() - startLoadTime;
    
    console.log(`Data loaded in ${loadTime.toFixed(2)}ms`);
    console.log(`Detection data size: ${formatBytes(Buffer.byteLength(JSON.stringify(detectionData)))}`);
    console.log(`Number of detections: ${detectionData.detections ? detectionData.detections.length : 'N/A'}`);
    
    // Calculate memory usage of segmentation data
    let segmentationCount = 0;
    let segmentationSize = 0;
    
    if (detectionData.detections && Array.isArray(detectionData.detections)) {
      for (const detection of detectionData.detections) {
        if (detection.segmentation) {
          segmentationCount++;
          segmentationSize += Buffer.byteLength(JSON.stringify(detection.segmentation));
        }
      }
    }
    
    console.log(`Detections with segmentation: ${segmentationCount}`);
    console.log(`Total segmentation data size: ${formatBytes(segmentationSize)}`);
    
    // Measure segmentation optimization potential
    console.log('\nSegmentation Optimization Potential:');
    let originalSegSize = 0;
    let optimizedSegSize = 0;
    
    if (detectionData.detections && detectionData.detections.length > 0) {
      const sampleDetections = detectionData.detections.slice(0, 10);
      
      for (const detection of sampleDetections) {
        if (detection.segmentation) {
          const originalSize = Buffer.byteLength(JSON.stringify(detection.segmentation));
          originalSegSize += originalSize;
          
          // Create optimized version (counts only)
          const optimized = {
            counts: detection.segmentation.counts,
            size: detection.segmentation.size
          };
          
          const optimizedSize = Buffer.byteLength(JSON.stringify(optimized));
          optimizedSegSize += optimizedSize;
          
          console.log(`  Sample segmentation: ${formatBytes(originalSize)} → ${formatBytes(optimizedSize)} (${((1 - optimizedSize/originalSize) * 100).toFixed(2)}% reduction)`);
        }
      }
      
      if (originalSegSize > 0) {
        const reduction = (1 - optimizedSegSize/originalSegSize) * 100;
        console.log(`  Average reduction: ${reduction.toFixed(2)}%`);
        console.log(`  Projected total savings: ${formatBytes(segmentationSize * (reduction/100))}`);
      }
    }
    
    // S2 Cell analysis
    console.log('\nS2 Cell Data:');
    if (detectionData.metadata && detectionData.metadata.s2_cells) {
      console.log(`  Center cell: ${detectionData.metadata.s2_cells.center_cell || 'N/A'}`);
      if (detectionData.metadata.s2_cells.mapping) {
        const mapping = detectionData.metadata.s2_cells.mapping;
        console.log(`  Mapping available: ${JSON.stringify(mapping, null, 2)}`);
        
        // Calculate precision based on mapping
        if (mapping.width && mapping.height && mapping.lng_per_pixel && mapping.lat_per_pixel) {
          const lngPrecision = mapping.lng_per_pixel * 111000; // approx meters per degree at equator
          const latPrecision = mapping.lat_per_pixel * 111000;
          console.log(`  Approximate precision: ${lngPrecision.toFixed(2)}m (longitude) × ${latPrecision.toFixed(2)}m (latitude)`);
        }
      } else {
        console.log('  No S2 cell mapping available');
      }
    } else {
      console.log('  No S2 cell data available');
    }
    
    // Final recommendations
    console.log('\nOptimization Recommendations:');
    console.log('  1. Reduce segmentation data size by storing only essential information');
    console.log('  2. Implement lazy loading for segmentation masks');
    console.log('  3. Use S2 cells for precise coordinate mapping');
    console.log('  4. Implement virtual rendering for objects outside viewport');
    console.log('  5. Batch DOM operations using document fragments');
    
  } catch (error) {
    console.error('Error during test:', error);
  }
}

// Helper function to format bytes
function formatBytes(bytes, decimals = 2) {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

runTest();