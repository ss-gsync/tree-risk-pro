// MLOverlay.js
//
// Enhanced implementation of ML Detection Overlay using Google Maps OverlayView
// for precise geographic alignment
//
// This provides a complete replacement for the DOM-based overlay with proper
// map positioning and automatic updating during zoom/pan operations

// Check if Google Maps API is available
const isGoogleMapsAvailable = () => {
  return typeof window !== 'undefined' && window.google && window.google.maps;
};

/**
 * MLDetectionOverlay - Custom Google Maps OverlayView implementation
 * for precise geographic positioning of detection boxes
 */
let MLDetectionOverlay;

// Conditionally define the MLDetectionOverlay class based on Google Maps availability
if (isGoogleMapsAvailable()) {
  // Only define the class if Google Maps is available
  MLDetectionOverlay = class MLDetectionOverlay extends google.maps.OverlayView {
    constructor(map, data, options) {
      super();
      this.map = map;
      this.data = data;
      this.options = options || {};
      this.setMap(map);
      
      // Calculate map bounds from data or use map's current bounds
      this.bounds = this.calculateBoundsFromData();
      console.log('MLDetectionOverlay: Created with bounds', this.bounds.toString());
    }
    
    /**
     * Called when the overlay is added to the map
     */
    onAdd() {
      // Create the main overlay container
      this.div = document.createElement('div');
      this.div.id = 'ml-detection-overlay';
      this.div.style.position = 'absolute';
      this.div.style.pointerEvents = 'none';
      this.div.style.opacity = this.options.opacity || 1.0;
      
      // Add to overlay pane
      const panes = this.getPanes();
      panes.overlayLayer.appendChild(this.div);
      
      // Create container for boxes
      this.boxesContainer = document.createElement('div');
      this.boxesContainer.id = 'detection-boxes-container';
      this.boxesContainer.style.position = 'absolute';
      this.boxesContainer.style.width = '100%';
      this.boxesContainer.style.height = '100%';
      this.boxesContainer.style.pointerEvents = 'none';
      this.div.appendChild(this.boxesContainer);
      
      // Add satellite tint if requested
      if (this.options.showTint) {
        this.addTintLayer(this.options.tintColor || 'rgba(0, 30, 60, 0.3)');
      }
      
      // Add job ID indicator if provided
      if (this.options.jobId) {
        this.addJobIdIndicator(this.options.jobId);
      }
      
      // Add count indicator
      this.addCountIndicator();
    }
    
    /**
     * Called when the overlay position should be updated
     */
    draw() {
      // Get projection for coordinate conversion
      const projection = this.getProjection();
      if (!projection) return;
      
      // Clear existing boxes
      this.boxesContainer.innerHTML = '';
      
      // Get current map bounds
      const mapBounds = this.map.getBounds();
      if (!mapBounds) return;
      
      // Create a document fragment for batch DOM operations
      const fragment = document.createDocumentFragment();
      
      // Define colors for tree risk categories
      const categoryColors = {
        healthy_tree: { base: '#16a34a', light: 'rgba(22, 163, 74, 0.15)', label: 'Healthy Tree' },
        hazardous_tree: { base: '#8b5cf6', light: 'rgba(139, 92, 246, 0.15)', label: 'Hazardous Tree' },
        dead_tree: { base: '#6b7280', light: 'rgba(107, 114, 128, 0.15)', label: 'Dead Tree' },
        low_canopy_tree: { base: '#0ea5e9', light: 'rgba(14, 165, 233, 0.15)', label: 'Low Canopy Tree' },
        pest_disease_tree: { base: '#84cc16', light: 'rgba(132, 204, 22, 0.15)', label: 'Pest/Disease Tree' },
        flood_prone_tree: { base: '#0891b2', light: 'rgba(8, 145, 178, 0.15)', label: 'Flood-Prone Tree' },
        utility_conflict_tree: { base: '#3b82f6', light: 'rgba(59, 130, 246, 0.15)', label: 'Utility Conflict Tree' },
        structural_hazard_tree: { base: '#0d9488', light: 'rgba(13, 148, 136, 0.15)', label: 'Structural Hazard Tree' },
        fire_risk_tree: { base: '#4f46e5', light: 'rgba(79, 70, 229, 0.15)', label: 'Fire Risk Tree' },
        building: { base: '#0ea5e9', light: 'rgba(14, 165, 233, 0.15)', label: 'Building' },
        power_line: { base: '#ef4444', light: 'rgba(239, 68, 68, 0.15)', label: 'Power Line' }
      };
      
      // Track rendering metrics
      const metrics = { 
        boxesRendered: 0, 
        segmentsRendered: 0,
        categoryCounts: {}
      };
      
      // Initialize category counts
      Object.keys(categoryColors).forEach(key => {
        metrics.categoryCounts[key] = 0;
      });
      
      // Process trees
      if (this.data.trees && this.data.trees.length > 0) {
        // Sort trees by area (larger first) for better visual layering
        const sortedTrees = [...this.data.trees].sort((a, b) => {
          const aArea = a.box.width * a.box.height;
          const bArea = b.box.width * b.box.height;
          return bArea - aArea;
        });
        
        // Limit to reasonable number for performance
        const treesToRender = sortedTrees.slice(0, 200);
        
        treesToRender.forEach((tree, index) => {
          if (!tree.box) return;
          
          // Determine tree category based on class or explicit category
          const category = this.determineTreeCategory(tree);
          
          // Skip if category is disabled in options
          if (this.options.classes && this.options.classes[category] === false) {
            return;
          }
          
          // Increment category count
          metrics.categoryCounts[category] = (metrics.categoryCounts[category] || 0) + 1;
          
          // Get colors for this category
          const color = categoryColors[category]?.base || '#16a34a';
          const bgColor = categoryColors[category]?.light || 'rgba(22, 163, 74, 0.15)';
          const label = categoryColors[category]?.label || 'Tree';
          
          // Calculate centroid from segmentation mask or bounding box
          const centroid = tree.segmentation ? 
            this.calculateMaskCentroid(tree.segmentation) : 
            { x: tree.box.x + tree.box.width/2, y: tree.box.y + tree.box.height/2 };
          
          // Convert normalized centroid to geographic coordinates
          const geoPoint = this.normalizedToLatLng(centroid.x, centroid.y);
          
          // Skip if outside current map view for performance
          if (!mapBounds.contains(geoPoint)) return;
          
          // Convert to pixel position using the projection
          const pixelPoint = projection.fromLatLngToDivPixel(geoPoint);
          
          // Calculate box size in pixels based on zoom
          const zoom = this.map.getZoom();
          const boxSize = this.calculateBoxSizeByZoom(tree.box.width, tree.box.height, zoom);
          
          // Create box element
          const box = document.createElement('div');
          box.className = `detection-box tree-box ${category}-box`;
          box.id = `tree-${index}`;
          box.style.position = 'absolute';
          box.style.left = `${pixelPoint.x - boxSize.width/2}px`;
          box.style.top = `${pixelPoint.y - boxSize.height/2}px`;
          box.style.width = `${boxSize.width}px`;
          box.style.height = `${boxSize.height}px`;
          box.style.border = `2px solid ${color}`;
          box.style.backgroundColor = bgColor;
          box.style.boxSizing = 'border-box';
          box.style.zIndex = '10';
          
          // Add data attributes for later reference
          box.dataset.type = 'tree';
          box.dataset.category = category;
          box.dataset.confidence = tree.confidence || '';
          box.dataset.lat = geoPoint.lat();
          box.dataset.lng = geoPoint.lng();
          
          // Add segmentation mask if available and enabled
          if (this.options.showSegmentation && tree.segmentation) {
            const mask = this.createSegmentationMask(
              tree.segmentation,
              color,
              boxSize.width,
              boxSize.height
            );
            
            if (mask) {
              box.appendChild(mask);
              metrics.segmentsRendered++;
            }
          }
          
          // Add label to some trees
          if (index % 5 === 0) {
            const confidence = tree.confidence ? ` ${Math.round(tree.confidence * 100)}%` : '';
            const labelElement = this.createLabelElement(`${label}${confidence}`, {
              backgroundColor: color,
              color: 'white'
            });
            box.appendChild(labelElement);
          }
          
          fragment.appendChild(box);
          metrics.boxesRendered++;
        });
      }
      
      // Process buildings if needed using same approach
      if (this.data.buildings && this.data.buildings.length > 0 && 
          (!this.options.classes || this.options.classes.buildings !== false)) {
        // Implement building rendering similar to trees
        // ...
      }
      
      // Process power lines if needed using same approach
      if (this.data.power_lines && this.data.power_lines.length > 0 && 
          (!this.options.classes || this.options.classes.power_lines !== false)) {
        // Implement power line rendering similar to trees
        // ...
      }
      
      // Add all boxes to container at once
      this.boxesContainer.appendChild(fragment);
      
      // Dispatch event with category counts
      window.dispatchEvent(new CustomEvent('detectionCategoryCounts', {
        detail: {
          counts: metrics.categoryCounts,
          total: metrics.boxesRendered
        }
      }));
      
      console.log(`MLOverlay: Rendered ${metrics.boxesRendered} boxes and ${metrics.segmentsRendered} masks at zoom ${this.map.getZoom()}`);
    }
    
    /**
     * Called when the overlay is removed from the map
     */
    onRemove() {
      if (this.div && this.div.parentNode) {
        this.div.parentNode.removeChild(this.div);
        this.div = null;
        this.boxesContainer = null;
      }
    }
    
    /**
     * Calculate geographic bounds from detection data
     */
    calculateBoundsFromData() {
      const mapInstance = this.map;
      
      // Try to get bounds from data metadata
      if (this.data.metadata && this.data.metadata.bounds) {
        const bounds = this.data.metadata.bounds;
        const sw = new google.maps.LatLng(bounds[0][1], bounds[0][0]);
        const ne = new google.maps.LatLng(bounds[1][1], bounds[1][0]);
        return new google.maps.LatLngBounds(sw, ne);
      }
      
      // If S2 cells are available, use them
      if (this.data.metadata && this.data.metadata.s2_cells && 
          this.data.metadata.s2_cells.cells && 
          this.data.metadata.s2_cells.cells.length > 0) {
        
        // Get the cell corners for the center cell
        if (this.data.metadata.s2_cells.center_bounds) {
          const cb = this.data.metadata.s2_cells.center_bounds;
          const sw = new google.maps.LatLng(cb[0][1], cb[0][0]);
          const ne = new google.maps.LatLng(cb[1][1], cb[1][0]);
          return new google.maps.LatLngBounds(sw, ne);
        }
      }
      
      // Fallback to map's current bounds
      if (mapInstance && mapInstance.getBounds) {
        return mapInstance.getBounds();
      }
      
      // Last resort - create a small bounds around map center
      const center = mapInstance.getCenter();
      const defaultBounds = new google.maps.LatLngBounds();
      defaultBounds.extend(new google.maps.LatLng(center.lat() - 0.01, center.lng() - 0.01));
      defaultBounds.extend(new google.maps.LatLng(center.lat() + 0.01, center.lng() + 0.01));
      return defaultBounds;
    }
    
    /**
     * Convert normalized coordinates to geographic coordinates
     */
    normalizedToLatLng(x, y) {
      // Use the calculated bounds
      const bounds = this.bounds;
      const sw = bounds.getSouthWest();
      const ne = bounds.getNorthEast();
      
      // Convert normalized [0-1] coordinates to geographic coordinates
      // Note the 1-y inversion for correct geographical orientation
      const lat = sw.lat() + (ne.lat() - sw.lat()) * (1 - y);
      const lng = sw.lng() + (ne.lng() - sw.lng()) * x;
      
      return new google.maps.LatLng(lat, lng);
    }
    
    /**
     * Calculate box size in pixels based on zoom level
     */
    calculateBoxSizeByZoom(normalizedWidth, normalizedHeight, zoom) {
      // Base size at zoom level 17 (typical for aerial imagery)
      const baseSizeAtZoom17 = 80;
      
      // Size multiplier based on zoom difference from base
      const zoomDiff = zoom - 17;
      const zoomFactor = Math.pow(2, zoomDiff);
      
      // Calculate aspect ratio
      const ratio = normalizedHeight / normalizedWidth;
      
      // Calculate dimensions with zoom adjustment
      const width = baseSizeAtZoom17 * zoomFactor;
      const height = width * ratio;
      
      return { width, height };
    }
    
    /**
     * Determine tree category based on class name or explicit category
     */
    determineTreeCategory(tree) {
      // Use explicit category if available
      if (tree.category && typeof tree.category === 'string') {
        return tree.category;
      }
      
      // Otherwise determine from class name
      const className = (tree.class || '').toLowerCase();
      
      if (className.includes('hazard') || className.includes('hazardous')) {
        return 'hazardous_tree';
      } else if (className.includes('dead')) {
        return 'dead_tree';
      } else if (className.includes('pest') || className.includes('disease')) {
        return 'pest_disease_tree';
      } else if (className.includes('low canopy')) {
        return 'low_canopy_tree';
      } else if (className.includes('flood prone')) {
        return 'flood_prone_tree';
      } else if (className.includes('utility conflict')) {
        return 'utility_conflict_tree';
      } else if (className.includes('structural hazard')) {
        return 'structural_hazard_tree';
      } else if (className.includes('fire risk')) {
        return 'fire_risk_tree';
      } else {
        return 'healthy_tree'; // Default
      }
    }
    
    /**
     * Calculate the centroid of a segmentation mask
     */
    calculateMaskCentroid(segmentation) {
      try {
        if (!segmentation || (!segmentation.counts && !segmentation.data)) {
          return { x: 0.5, y: 0.5 }; // Default to center
        }
        
        const width = segmentation.size[1];
        const height = segmentation.size[0];
        
        // Process RLE data into a binary mask if needed
        let maskData;
        if (segmentation.counts) {
          maskData = this.decodeMaskRLE(segmentation);
        } else if (segmentation.data) {
          maskData = new Uint8Array(segmentation.data);
        } else {
          return { x: 0.5, y: 0.5 }; // Default to center
        }
        
        // Calculate centroid from mask data
        let totalX = 0, totalY = 0;
        let count = 0;
        
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            if (maskData[idx]) {
              totalX += x;
              totalY += y;
              count++;
            }
          }
        }
        
        if (count === 0) {
          return { x: 0.5, y: 0.5 }; // Default to center
        }
        
        // Return normalized coordinates
        return {
          x: totalX / count / width,
          y: totalY / count / height
        };
      } catch (error) {
        console.error('Error calculating mask centroid:', error);
        return { x: 0.5, y: 0.5 }; // Default to center
      }
    }
    
    /**
     * Decode RLE format mask
     */
    decodeMaskRLE(segmentation) {
      try {
        const counts = segmentation.counts;
        const h = segmentation.size[0];
        const w = segmentation.size[1];
        const size = h * w;
        
        // Initialize mask array
        const mask = new Uint8Array(size);
        
        // Process RLE data differently based on format
        if (typeof counts === 'string') {
          // Binary string format
          let pos = 0;
          let fill = 0; // Start with background (0)
          
          for (let i = 0; i < counts.length; i++) {
            const count = counts.charCodeAt(i);
            // Use efficient array filling
            mask.fill(fill, pos, pos + count);
            pos += count;
            fill = 1 - fill; // Toggle between 0 and 1
          }
        } else if (Array.isArray(counts)) {
          // Array format
          let pos = 0;
          let fill = 0; // Start with background (0)
          
          for (let i = 0; i < counts.length; i++) {
            const count = counts[i];
            // Use efficient array filling
            mask.fill(fill, pos, pos + count);
            pos += count;
            fill = 1 - fill; // Toggle between 0 and 1
          }
        }
        
        return mask;
      } catch (error) {
        console.error('Error decoding mask RLE:', error);
        return null;
      }
    }
    
    /**
     * Create a segmentation mask with canvas
     */
    createSegmentationMask(segmentation, color, width, height) {
      try {
        // Skip if missing required data
        if (!segmentation || !segmentation.size || (!segmentation.counts && !segmentation.data)) {
          return null;
        }
        
        // Extract RGB components from color string
        let r = 40, g = 167, b = 69; // Default green
        const hexMatch = color.match(/#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})/i);
        const rgbaMatch = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*[\d.]+)?\)/);
        
        if (hexMatch) {
          r = parseInt(hexMatch[1], 16);
          g = parseInt(hexMatch[2], 16);
          b = parseInt(hexMatch[3], 16);
        } else if (rgbaMatch) {
          r = parseInt(rgbaMatch[1], 10);
          g = parseInt(rgbaMatch[2], 10);
          b = parseInt(rgbaMatch[3], 10);
        }
        
        // Create canvas for rendering the mask
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.pointerEvents = 'none';
        canvas.style.opacity = '0.6';
        canvas.className = 'segmentation-mask';
        
        // Get the drawing context
        const ctx = canvas.getContext('2d');
        
        // Create ImageData
        const imageData = ctx.createImageData(canvas.width, canvas.height);
        
        // Convert segmentation format to binary mask
        let mask;
        
        // Handle different mask formats
        if (segmentation.counts) {
          // RLE format
          mask = this.decodeMaskRLE(segmentation);
        } else if (segmentation.data) {
          // Direct bitmap format
          mask = new Uint8Array(segmentation.data);
        } else if (Array.isArray(segmentation)) {
          // Raw 2D array format
          mask = new Uint8Array(canvas.width * canvas.height);
          let idx = 0;
          for (let row = 0; row < segmentation.length; row++) {
            for (let col = 0; col < segmentation[row].length; col++) {
              mask[idx++] = segmentation[row][col] ? 1 : 0;
            }
          }
        }
        
        if (!mask) return null;
        
        // Scale mask to fit the canvas
        const scaledMask = this.scaleMaskToFit(
          mask, 
          segmentation.size[1], // original width
          segmentation.size[0], // original height
          canvas.width, 
          canvas.height
        );
        
        // Use Uint32Array for faster processing
        const buf = new ArrayBuffer(imageData.data.length);
        const buf8 = new Uint8ClampedArray(buf);
        const data32 = new Uint32Array(buf);
        
        // Pre-compute the pixel value (ABGR format for little-endian)
        const abgr = (255 << 24) | // alpha
                     (b << 16) |    // blue
                     (g << 8) |     // green
                     r;             // red
        
        // Fast filling of image data
        const pixelCount = scaledMask.length;
        for (let i = 0; i < pixelCount; i++) {
          if (scaledMask[i]) {
            data32[i] = abgr;
          }
        }
        
        // Copy buffer back to imageData
        imageData.data.set(buf8);
        
        // Put the image data on the canvas
        ctx.putImageData(imageData, 0, 0);
        
        return canvas;
      } catch (error) {
        console.error('Error creating segmentation mask:', error);
        return null;
      }
    }
    
    /**
     * Scale a mask to fit the target dimensions
     */
    scaleMaskToFit(mask, srcWidth, srcHeight, destWidth, destHeight) {
      const scaledMask = new Uint8Array(destWidth * destHeight);
      
      const xRatio = srcWidth / destWidth;
      const yRatio = srcHeight / destHeight;
      
      for (let y = 0; y < destHeight; y++) {
        for (let x = 0; x < destWidth; x++) {
          // Get corresponding position in source mask
          const srcX = Math.floor(x * xRatio);
          const srcY = Math.floor(y * yRatio);
          const srcIdx = srcY * srcWidth + srcX;
          
          // Set value in destination mask
          const destIdx = y * destWidth + x;
          scaledMask[destIdx] = mask[srcIdx];
        }
      }
      
      return scaledMask;
    }
    
    /**
     * Create a label element for a detection box
     */
    createLabelElement(text, options = {}) {
      const styles = {
        backgroundColor: options.backgroundColor || 'rgba(40, 167, 69, 0.9)',
        color: options.color || 'white'
      };
      
      const label = document.createElement('div');
      label.style.position = 'absolute';
      label.style.top = '-20px';
      label.style.left = '0';
      label.style.backgroundColor = styles.backgroundColor;
      label.style.color = styles.color;
      label.style.padding = '2px 6px';
      label.style.fontSize = '11px';
      label.style.fontWeight = 'bold';
      label.style.borderRadius = '4px';
      label.style.whiteSpace = 'nowrap';
      label.style.zIndex = '11';
      label.textContent = text;
      
      return label;
    }
    
    /**
     * Add count indicator to the overlay
     */
    addCountIndicator() {
      const data = this.data;
      const treeCount = data.trees?.length || 0;
      const buildingCount = data.buildings?.length || 0;
      const powerLineCount = data.power_lines?.length || 0;
      const totalCount = treeCount + buildingCount + powerLineCount;
      
      const counter = document.createElement('div');
      counter.style.position = 'absolute';
      counter.style.bottom = '20px';
      counter.style.right = '20px';
      counter.style.backgroundColor = 'rgba(33, 33, 33, 0.85)';
      counter.style.color = 'white';
      counter.style.padding = '8px 12px';
      counter.style.borderRadius = '6px';
      counter.style.fontSize = '13px';
      counter.style.zIndex = '11';
      counter.style.cursor = 'pointer';
      
      counter.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <span>${totalCount} objects detected</span>
          <button style="background:none;border:none;color:white;font-size:12px;margin-left:8px;cursor:pointer;">▼</button>
        </div>
        <div style="display:none;margin-top:8px;border-top:1px solid rgba(255,255,255,0.1);padding-top:8px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="color:#4ade80;">Trees:</span>
            <span>${treeCount}</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="color:#60a5fa;">Buildings:</span>
            <span>${buildingCount}</span>
          </div>
          <div style="display:flex;justify-content:space-between;">
            <span style="color:#fbbf24;">Power Lines:</span>
            <span>${powerLineCount}</span>
          </div>
        </div>
      `;
      
      // Toggle details when clicked
      const details = counter.querySelector('div:nth-child(2)');
      const toggle = counter.querySelector('button');
      
      counter.addEventListener('click', () => {
        const isVisible = details.style.display !== 'none';
        details.style.display = isVisible ? 'none' : 'block';
        toggle.textContent = isVisible ? '▼' : '▲';
      });
      
      this.div.appendChild(counter);
    }
    
    /**
     * Add a tint layer to the overlay (for satellite imagery mode)
     */
    addTintLayer(tintColor) {
      const tintLayer = document.createElement('div');
      tintLayer.className = 'tint-layer';
      tintLayer.style.position = 'absolute';
      tintLayer.style.top = '0';
      tintLayer.style.left = '0';
      tintLayer.style.width = '100%';
      tintLayer.style.height = '100%';
      tintLayer.style.backgroundColor = tintColor;
      tintLayer.style.pointerEvents = 'none';
      tintLayer.style.zIndex = '1';
      
      const indicator = document.createElement('div');
      indicator.style.position = 'absolute';
      indicator.style.top = '10px';
      indicator.style.right = '10px';
      indicator.style.padding = '4px 8px';
      indicator.style.fontSize = '11px';
      indicator.style.backgroundColor = 'rgba(0,0,0,0.6)';
      indicator.style.color = 'white';
      indicator.style.borderRadius = '4px';
      indicator.textContent = 'Satellite Mode';
      
      tintLayer.appendChild(indicator);
      this.div.appendChild(tintLayer);
    }
    
    /**
     * Add job ID indicator to the overlay
     */
    addJobIdIndicator(jobId) {
      const jobInfo = document.createElement('div');
      jobInfo.style.position = 'absolute';
      jobInfo.style.bottom = '10px';
      jobInfo.style.left = '10px';
      jobInfo.style.padding = '4px 8px';
      jobInfo.style.fontSize = '11px';
      jobInfo.style.backgroundColor = 'rgba(0,0,0,0.6)';
      jobInfo.style.color = 'white';
      jobInfo.style.borderRadius = '4px';
      jobInfo.textContent = `Job ID: ${jobId || 'N/A'}`;
      this.div.appendChild(jobInfo);
    }
    
    /**
     * Update the overlay opacity - enhanced for comprehensive updates
     */
    updateOpacity(opacity) {
      console.log(`MLDetectionOverlay: updateOpacity called with ${opacity}`);
      
      // Update class instance options
      this.options.opacity = opacity;
      
      // Update main div opacity
      if (this.div) {
        this.div.style.opacity = opacity;
      }
      
      // Update any segmentation masks inside this overlay
      try {
        const masks = this.div?.querySelectorAll('.segmentation-mask') || [];
        if (masks.length > 0) {
          masks.forEach(mask => {
            mask.style.opacity = Math.min(opacity * 1.2, 0.8); // Slightly adjust for better visibility
          });
          console.log(`MLDetectionOverlay: Updated opacity on ${masks.length} segmentation masks`);
        }
      } catch (e) {
        console.error('Error updating segmentation mask opacity:', e);
      }
      
      // Update any tint layers
      try {
        const tintLayers = this.div?.querySelectorAll('.tint-layer') || [];
        if (tintLayers.length > 0) {
          tintLayers.forEach(layer => {
            layer.style.opacity = opacity;
          });
          console.log(`MLDetectionOverlay: Updated opacity on ${tintLayers.length} tint layers`);
        }
      } catch (e) {
        console.error('Error updating tint layer opacity:', e);
      }
      
      // Update global settings for consistency
      window.mlOverlaySettings = {
        ...(window.mlOverlaySettings || {}),
        opacity: opacity
      };
      
      // Dispatch event for other components
      try {
        window.dispatchEvent(new CustomEvent('mlOverlayOpacityUpdated', {
          detail: { 
            opacity: opacity,
            source: 'ml_detection_overlay_class',
            timestamp: Date.now()
          }
        }));
      } catch (e) {
        console.error('Error dispatching opacity update event:', e);
      }
      
      return true;
    }
    
    /**
     * Update the class visibility
     */
    updateClasses(classes) {
      this.options.classes = classes;
      this.draw(); // Redraw with updated class visibility
    }
  };
} else {
  // Create a placeholder class if Google Maps isn't available yet
  MLDetectionOverlay = class MLDetectionOverlay {
    constructor(map, data, options) {
      console.warn('Google Maps API not available yet. The overlay will not be fully functional.');
      this.map = map;
      this.data = data;
      this.options = options || {};
    }
    
    // Placeholder methods that will be defined properly once Google Maps loads
    setMap() {}
    onAdd() {}
    draw() {}
    onRemove() {}
    calculateBoundsFromData() { return null; }
    normalizedToLatLng() { return null; }
    calculateBoxSizeByZoom() { return {width: 0, height: 0}; }
    determineTreeCategory() { return 'unknown'; }
    calculateMaskCentroid() { return {x: 0.5, y: 0.5}; }
    decodeMaskRLE() { return null; }
    createSegmentationMask() { return null; }
    scaleMaskToFit() { return new Uint8Array(0); }
    createLabelElement() { return document.createElement('div'); }
    addCountIndicator() {}
    addTintLayer() {}
    addJobIdIndicator() {}
    updateOpacity() {}
    updateClasses() {}
  };
}

/**
 * Function to create and add a new overlay
 * @param {Object} map - Google Maps instance
 * @param {Object} data - Detection data with trees array
 * @param {Object} options - Configuration options for the overlay
 * @returns {MLDetectionOverlay|null} - The created overlay or null if Google Maps is not available
 */
function renderMLOverlay(map, data, options = {}) {
  if (!isGoogleMapsAvailable()) {
    console.warn('renderMLOverlay: Google Maps API not available yet. Deferring overlay creation.');
    
    // Set up a watch to create the overlay once Google Maps is available
    const checkInterval = setInterval(() => {
      if (isGoogleMapsAvailable()) {
        clearInterval(checkInterval);
        console.log('renderMLOverlay: Google Maps API now available, creating overlay');
        renderMLOverlay(map, data, options);
      }
    }, 500);
    
    // Clear interval after 10 seconds to prevent memory leaks
    setTimeout(() => clearInterval(checkInterval), 10000);
    return null;
  }

  // Process options with defaults
  const opts = {
    opacity: options.opacity !== undefined ? options.opacity : 1.0,
    classes: options.classes || {
      // Updated tree risk categories
      healthy_tree: true,
      hazardous_tree: true,
      dead_tree: true,
      low_canopy_tree: true,
      pest_disease_tree: true,
      flood_prone_tree: true,
      utility_conflict_tree: true,
      structural_hazard_tree: true,
      fire_risk_tree: true
    },
    showTint: options.showTint !== undefined ? options.showTint : false,
    showSegmentation: options.showSegmentation !== undefined ? options.showSegmentation : true,
    jobId: options.jobId || null,
    appendMode: options.appendMode !== undefined ? options.appendMode : false,
    tintColor: options.tintColor || 'rgba(0, 30, 60, 0.3)'
  };
  
  // Remove existing overlay if not in append mode
  if (!opts.appendMode && window._mlDetectionOverlay) {
    window._mlDetectionOverlay.setMap(null);
    window._mlDetectionOverlay = null;
  }
  
  try {
    // Create a new overlay
    const overlay = new MLDetectionOverlay(map, data, opts);
    
    // Store reference globally for later access
    window._mlDetectionOverlay = overlay;
    
    // Update global settings for consistency with other components
    window.mlOverlaySettings = window.mlOverlaySettings || {};
    window.mlOverlaySettings.showOverlay = true;
    window.mlOverlaySettings.showSegmentation = opts.showSegmentation;
    window.mlOverlaySettings.opacity = opts.opacity;
    window.mlOverlaySettings.showCategories = opts.classes;
    
    console.log('MLOverlay: Overlay added to map');
    return overlay;
  } catch (error) {
    console.error('MLOverlay: Failed to create overlay', error);
    return null;
  }
}

/**
 * Function to remove existing overlay
 */
function removeMLOverlay() {
  if (window._mlDetectionOverlay) {
    window._mlDetectionOverlay.setMap(null);
    window._mlDetectionOverlay = null;
    return true;
  }
  return false;
}

/**
 * Function to update opacity - enhanced for real-time updates
 */
function updateMLOverlayOpacity(opacity) {
  console.log(`MLOverlay: updateMLOverlayOpacity called with ${opacity}`);
  
  // Update global settings for other components
  window.mlOverlaySettings = {
    ...(window.mlOverlaySettings || {}),
    opacity: opacity
  };
  
  // Save to localStorage for persistence
  try {
    localStorage.setItem('ml-overlay-opacity', opacity.toString());
  } catch (e) {
    console.error("Error saving opacity to localStorage:", e);
  }
  
  // APPROACH 1: Use MLDetectionOverlay instance if available (most efficient)
  if (window._mlDetectionOverlay) {
    try {
      // Use the dedicated method if available
      if (typeof window._mlDetectionOverlay.updateOpacity === 'function') {
        window._mlDetectionOverlay.updateOpacity(opacity);
        console.log(`MLOverlay: Updated opacity using _mlDetectionOverlay.updateOpacity(${opacity})`);
        return true;
      }
      
      // Direct DOM update if method not available but div exists
      if (window._mlDetectionOverlay.div) {
        window._mlDetectionOverlay.div.style.opacity = opacity;
        console.log(`MLOverlay: Updated opacity directly on div: ${opacity}`);
        return true;
      }
    } catch (e) {
      console.error('Error updating MLDetectionOverlay opacity:', e);
    }
  }
  
  // APPROACH 2: Direct DOM manipulation as fallback
  const overlay = document.getElementById('ml-detection-overlay');
  if (overlay) {
    overlay.style.opacity = opacity;
    console.log(`MLOverlay: Updated opacity on DOM element: ${opacity}`);
    return true;
  }
  
  // APPROACH 3: Update tint layers if they exist
  const tintLayers = document.querySelectorAll('.tint-layer');
  if (tintLayers.length > 0) {
    let success = false;
    tintLayers.forEach(layer => {
      layer.style.opacity = opacity;
      success = true;
    });
    
    if (success) {
      console.log(`MLOverlay: Updated opacity on ${tintLayers.length} tint layers: ${opacity}`);
      return true;
    }
  }
  
  // APPROACH 4: Update segmentation masks if they exist
  const masks = document.querySelectorAll('.segmentation-mask');
  if (masks.length > 0) {
    masks.forEach(mask => {
      mask.style.opacity = opacity;
    });
    console.log(`MLOverlay: Updated opacity on ${masks.length} segmentation masks: ${opacity}`);
    return true;
  }
  
  console.warn('MLOverlay: Could not find any elements to update opacity');
  return false;
}

/**
 * Function to update class visibility
 */
function updateMLOverlayClasses(classes) {
  if (window._mlDetectionOverlay) {
    window._mlDetectionOverlay.updateClasses(classes);
    return true;
  }
  return false;
}

/**
 * Function to toggle overlay visibility
 */
function toggleMLOverlayVisibility(visible) {
  console.log(`MLOverlay: toggleMLOverlayVisibility called with visible=${visible}`);
  
  if (window._mlDetectionOverlay) {
    try {
      // If we have the overlay instance, use it directly
      if (window._mlDetectionOverlay.div) {
        window._mlDetectionOverlay.div.style.display = visible ? 'block' : 'none';
        
        // Update global settings for consistency
        window.mlOverlaySettings = window.mlOverlaySettings || {};
        window.mlOverlaySettings.showOverlay = visible;
        window.detectionShowOverlay = visible;
        
        return true;
      }
    } catch (e) {
      console.error('Error in toggleMLOverlayVisibility:', e);
    }
  }
  
  // Direct DOM approach as fallback
  const overlay = document.getElementById('ml-detection-overlay');
  if (overlay) {
    overlay.style.display = visible ? 'block' : 'none';
    
    // Update global settings for consistency
    window.mlOverlaySettings = window.mlOverlaySettings || {};
    window.mlOverlaySettings.showOverlay = visible;
    window.detectionShowOverlay = visible;
    
    return true;
  }
  
  return false;
}

// Ensure initialization function
function ensureInitialized() {
  // Check if Google Maps API is available
  if (!isGoogleMapsAvailable()) {
    console.warn('ensureInitialized: Google Maps API not available yet.');
    return false;
  }
  
  console.log('MLOverlay: ensureInitialized called, Google Maps API is available');
  return true;
}

// Export overlay class and helper functions
export { 
  MLDetectionOverlay,
  renderMLOverlay,
  removeMLOverlay,
  updateMLOverlayOpacity,
  updateMLOverlayClasses,
  toggleMLOverlayVisibility,
  ensureInitialized
};

// Add to global scope for script interop
window.MLDetectionOverlay = MLDetectionOverlay;
window.renderMLOverlay = renderMLOverlay;
window.removeMLOverlay = removeMLOverlay;
window.updateMLOverlayOpacity = updateMLOverlayOpacity;
window.updateMLOverlayClasses = updateMLOverlayClasses;
window.toggleMLOverlayVisibility = toggleMLOverlayVisibility;
window.ensureMLOverlayInitialized = ensureInitialized;