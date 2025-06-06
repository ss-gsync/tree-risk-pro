// MLOverlay.js
//
// Enhanced implementation of ML Detection Overlay using Google Maps OverlayView
// for precise geographic alignment
//
// This implementation:
// 1. Uses Google Maps OverlayView for proper geographic positioning
// 2. Calculates segmentation mask centroids for precise alignment
// 3. Automatically updates during map zoom/pan operations
// 4. Handles tree risk categories consistently with the dashboard

/**
 * MLDetectionOverlay - Custom Google Maps OverlayView implementation
 * for precise geographic positioning of detection boxes
 * 
 * IMPORTANT: This class requires Google Maps API to be fully loaded
 * before it can be instantiated or used.
 */
let MLDetectionOverlay;

// Create the overlay class only when Google Maps is available
const initMLOverlayClass = () => {
  if (typeof google === 'undefined' || !google.maps || !google.maps.OverlayView) {
    console.warn('Google Maps API not loaded yet, MLDetectionOverlay will be initialized when ready');
    return false;
  }
  
  // Only define the class once
  if (MLDetectionOverlay) return true;
  
  // Define the overlay class
  MLDetectionOverlay = class MLDetectionOverlay extends google.maps.OverlayView {
  constructor(map, data, options) {
    super();
    this.map = map;
    this.data = data;
    this.options = options || {};
    
    // Calculate map bounds from data or use map's current bounds
    this.bounds = this.calculateBoundsFromData();
    console.log('MLDetectionOverlay: Created with bounds', this.bounds.toString());
    
    // Setup throttled redraw for better performance
    this._setupThrottledEvents();
    
    // Add to map
    this.setMap(map);
  }
  
  /**
   * Setup throttled event handlers for map pan/zoom events
   * @private
   */
  _setupThrottledEvents() {
    // Variable to store timeout ID
    this._drawTimeoutId = null;
    this._lastDrawTime = 0;
    this._pendingDraw = false;
    
    // Throttle time in milliseconds
    this.throttleTime = 100; // Adjust based on performance needs
    
    // Throttled draw function for smooth panning/zooming
    this.throttledDraw = () => {
      // Cancel any pending draw
      if (this._drawTimeoutId) {
        clearTimeout(this._drawTimeoutId);
        this._drawTimeoutId = null;
      }
      
      // Check if we should draw now or schedule for later
      const now = performance.now();
      const timeSinceLastDraw = now - this._lastDrawTime;
      
      if (!this._pendingDraw && timeSinceLastDraw >= this.throttleTime) {
        // Enough time has passed since last draw, do it now
        this._lastDrawTime = now;
        this.draw();
      } else {
        // Schedule a draw for later
        this._pendingDraw = true;
        this._drawTimeoutId = setTimeout(() => {
          this._lastDrawTime = performance.now();
          this._pendingDraw = false;
          this.draw();
        }, this.throttleTime);
      }
    };
    
    // Register event listeners if map is available
    if (this.map) {
      // Track these events to trigger redraws when the map changes
      google.maps.event.addListener(this.map, 'idle', this.throttledDraw);
      google.maps.event.addListener(this.map, 'zoom_changed', this.throttledDraw);
      
      // Optional: special handling for drag events to minimize performance impact
      google.maps.event.addListener(this.map, 'dragend', this.throttledDraw);
      google.maps.event.addListener(this.map, 'dragstart', () => {
        // Reduce opacity during drag for better performance
        if (this.div) {
          this.div.style.opacity = '0.6';
        }
      });
      
      google.maps.event.addListener(this.map, 'drag', () => {
        // Only update positions during drag, don't fully redraw
        this._updatePositionsOnly();
      });
    }
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
    
    // Count indicator removed as requested
  }
  
  /**
   * Called when the overlay position should be updated
   */
  draw() {
    // Get projection for coordinate conversion
    const projection = this.getProjection();
    if (!projection) return;
    
    // Get current map bounds
    const mapBounds = this.map.getBounds();
    if (!mapBounds) return;
    
    // Initialize object pool if needed
    this._initializeObjectPool();
    
    // Clear visibility on all pooled objects
    this._resetObjectPool();
    
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
      boxesSkipped: 0,
      boxesFromPool: 0,
      categoryCounts: {}
    };
    
    // Initialize category counts
    Object.keys(categoryColors).forEach(key => {
      metrics.categoryCounts[key] = 0;
    });
    
    // Get visible boundaries of the map viewport in pixel coordinates
    const ne = mapBounds.getNorthEast();
    const sw = mapBounds.getSouthWest();
    const nePx = projection.fromLatLngToDivPixel(ne);
    const swPx = projection.fromLatLngToDivPixel(sw);
    
    // Add a buffer around the viewport for smoother panning
    const bufferPx = 200; // Buffer in pixels
    const viewportBounds = {
      min: { x: swPx.x - bufferPx, y: nePx.y - bufferPx },
      max: { x: nePx.x + bufferPx, y: swPx.y + bufferPx }
    };
    
    // Get current zoom level - for deciding rendering strategy
    const zoom = this.map.getZoom();
    
    // Determine maximum trees to render based on zoom level
    // Render more trees at higher zoom levels
    let maxTreesToRender = 200;
    if (zoom >= 19) maxTreesToRender = 300;
    else if (zoom >= 18) maxTreesToRender = 250;
    else if (zoom <= 15) maxTreesToRender = 150;
    else if (zoom <= 14) maxTreesToRender = 100;
    
    // Process trees
    if (this.data.trees && this.data.trees.length > 0) {
      // Filter and sort trees
      // First filter trees to ensure we only process those with valid box property
      const validTrees = (this.data.trees || []).filter(tree => tree && tree.box && 
        typeof tree.box === 'object' && 
        typeof tree.box.width === 'number' && 
        typeof tree.box.height === 'number');
      
      // Then sort the valid trees by area (larger first for better layering)
      const sortedTrees = [...validTrees].sort((a, b) => {
        const aArea = a.box.width * a.box.height;
        const bArea = b.box.width * b.box.height;
        return bArea - aArea;
      });
      
      // Optimize by only processing trees that could be in the viewport
      // For large datasets, we'll do a rough filtering first
      let treesToProcess = sortedTrees;
      if (sortedTrees.length > 500) {
        // Quick filter: check which trees are likely in viewport
        treesToProcess = sortedTrees.filter(tree => {
          if (!tree.box) return false;
          
          // Calculate centroid
          let centroidX, centroidY;
          if (tree.segmentation) {
            const calculatedCentroid = this.calculateMaskCentroid(tree.segmentation);
            centroidX = calculatedCentroid.x;
            centroidY = calculatedCentroid.y;
          } else if (tree.box) {
            centroidX = tree.box.x + tree.box.width/2;
            centroidY = tree.box.y + tree.box.height/2;
          } else if (tree.bbox && tree.bbox.length === 4) {
            centroidX = (tree.bbox[0] + tree.bbox[2]) / 2;
            centroidY = (tree.bbox[1] + tree.bbox[3]) / 2;
          } else {
            centroidX = 0.5;
            centroidY = 0.5;
          }
          
          // Convert to geo coordinates
          const geoPoint = this.normalizedToLatLng(centroidX, centroidY);
          
          // Quick check if in viewport (avoids expensive pixel projection)
          return mapBounds.contains(geoPoint);
        });
      }
      
      // Limit to reasonable number for performance
      const treesToRender = treesToProcess.slice(0, maxTreesToRender);
      
      metrics.boxesSkipped = sortedTrees.length - treesToRender.length;
      
      // Render trees using object pooling
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
        let centroid;
        if (tree.segmentation) {
          centroid = this.calculateMaskCentroid(tree.segmentation);
        } else if (tree.box && typeof tree.box === 'object' && 
                  typeof tree.box.x === 'number' && 
                  typeof tree.box.width === 'number' &&
                  typeof tree.box.y === 'number' &&
                  typeof tree.box.height === 'number') {
          centroid = { x: tree.box.x + tree.box.width/2, y: tree.box.y + tree.box.height/2 };
        } else if (tree.bbox && Array.isArray(tree.bbox) && tree.bbox.length === 4) {
          // Try to get centroid from bbox array [x1, y1, x2, y2]
          centroid = { 
            x: (tree.bbox[0] + tree.bbox[2]) / 2, 
            y: (tree.bbox[1] + tree.bbox[3]) / 2 
          };
        } else {
          // Default to center of the image if no valid coordinates
          centroid = { x: 0.5, y: 0.5 };
        }
        
        // Convert normalized centroid to geographic coordinates
        const geoPoint = this.normalizedToLatLng(centroid.x, centroid.y);
        
        // Skip if outside current map view for performance (with buffer)
        if (!mapBounds.contains(geoPoint)) return;
        
        // Convert to pixel position using the projection
        const pixelPoint = projection.fromLatLngToDivPixel(geoPoint);
        
        // Skip if definitely outside viewport with buffer
        if (pixelPoint.x < viewportBounds.min.x || pixelPoint.x > viewportBounds.max.x ||
            pixelPoint.y < viewportBounds.min.y || pixelPoint.y > viewportBounds.max.y) {
          return;
        }
        
        // Calculate box size in pixels based on zoom
        let boxWidth, boxHeight;
        
        // Determine box dimensions from available properties
        if (tree.box && typeof tree.box === 'object' && 
            typeof tree.box.width === 'number' && 
            typeof tree.box.height === 'number') {
          boxWidth = tree.box.width;
          boxHeight = tree.box.height;
        } else if (tree.bbox && Array.isArray(tree.bbox) && tree.bbox.length === 4) {
          // Calculate from bbox array [x1, y1, x2, y2]
          boxWidth = tree.bbox[2] - tree.bbox[0];
          boxHeight = tree.bbox[3] - tree.bbox[1];
        } else {
          // Default box size if dimensions not available
          boxWidth = 0.05;
          boxHeight = 0.05;
        }
        
        const boxSize = this.calculateBoxSizeByZoom(boxWidth, boxHeight, zoom);
        
        // Get or create box element from object pool
        const box = this._getBoxFromPool(category, tree.id || index);
        metrics.boxesFromPool++;
        
        // Update box attributes
        box.className = `detection-box tree-box ${category}-box`;
        box.id = `tree-${tree.id || index}`;
        box.style.position = 'absolute';
        box.style.left = `${pixelPoint.x - boxSize.width/2}px`;
        box.style.top = `${pixelPoint.y - boxSize.height/2}px`;
        box.style.width = `${boxSize.width}px`;
        box.style.height = `${boxSize.height}px`;
        box.style.border = `2px solid ${color}`;
        box.style.backgroundColor = bgColor;
        box.style.boxSizing = 'border-box';
        box.style.zIndex = '10';
        box.style.display = 'block'; // Ensure visibility
        
        // Add data attributes for later reference
        box.dataset.type = 'tree';
        box.dataset.category = category;
        box.dataset.confidence = tree.confidence || '';
        box.dataset.lat = geoPoint.lat();
        box.dataset.lng = geoPoint.lng();
        
        // Add segmentation mask if available and enabled
        if (this.options.showSegmentation && tree.segmentation) {
          // Get or create mask from pool
          const mask = this._getSegmentationMaskFromPool(tree.id || index);
          
          // Update mask with current data
          this._updateSegmentationMask(
            mask,
            tree.segmentation,
            color,
            boxSize.width,
            boxSize.height
          );
          
          // Add to box if not already there
          if (mask.parentNode !== box) {
            box.appendChild(mask);
          }
          
          metrics.segmentsRendered++;
        } else {
          // Remove any existing mask if segmentation is disabled
          const existingMask = box.querySelector('.segmentation-mask');
          if (existingMask) existingMask.style.display = 'none';
        }
        
        // Add label to some trees (using modulo to limit frequency)
        if (index % 5 === 0) {
          // Get or create label from pool
          const labelElement = this._getLabelFromPool(tree.id || index);
          
          // Update label content
          const confidence = tree.confidence ? ` ${Math.round(tree.confidence * 100)}%` : '';
          labelElement.textContent = `${label}${confidence}`;
          labelElement.style.backgroundColor = color;
          labelElement.style.color = 'white';
          
          // Add to box if not already there
          if (labelElement.parentNode !== box) {
            box.appendChild(labelElement);
          }
        } else {
          // Hide label if not needed for this tree
          const existingLabel = box.querySelector('.detection-label');
          if (existingLabel) existingLabel.style.display = 'none';
        }
        
        // No need to add to fragment - element is already in the DOM from the pool
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
    
    // Dispatch event with category counts
    window.dispatchEvent(new CustomEvent('detectionCategoryCounts', {
      detail: {
        counts: metrics.categoryCounts,
        total: metrics.boxesRendered
      }
    }));
    
    // Dispatch draw complete event for performance monitoring
    window.dispatchEvent(new CustomEvent('mlOverlayDrawComplete', {
      detail: {
        boxes: metrics.boxesRendered,
        fromPool: metrics.boxesFromPool,
        skipped: metrics.boxesSkipped,
        segmentation: metrics.segmentsRendered,
        zoom: this.map.getZoom(),
        timestamp: performance.now()
      }
    }));
    
    console.log(`MLOverlay: Rendered ${metrics.boxesRendered} boxes (${metrics.boxesFromPool} from pool) and ${metrics.segmentsRendered} masks at zoom ${this.map.getZoom()}, skipped ${metrics.boxesSkipped}`);
  }
  
  /**
   * Initialize object pools for DOM elements
   * @private
   */
  _initializeObjectPool() {
    // Initialize if needed
    if (!this._objectPool) {
      this._objectPool = {
        boxes: new Map(),
        masks: new Map(),
        labels: new Map()
      };
    }
  }
  
  /**
   * Reset the object pool by hiding all elements
   * @private
   */
  _resetObjectPool() {
    // Hide all boxes in pool
    this._objectPool?.boxes.forEach(box => {
      box.style.display = 'none';
    });
    
    // Hide all masks in pool
    this._objectPool?.masks.forEach(mask => {
      mask.style.display = 'none';
    });
    
    // Hide all labels in pool
    this._objectPool?.labels.forEach(label => {
      label.style.display = 'none';
    });
  }
  
  /**
   * Get a box element from the object pool or create a new one
   * @private
   * @param {string} category - The tree category
   * @param {string|number} id - Unique identifier for the box
   * @returns {HTMLElement} - The box element
   */
  _getBoxFromPool(category, id) {
    const key = `${category}-${id}`;
    
    // Check if element exists in pool
    if (this._objectPool.boxes.has(key)) {
      return this._objectPool.boxes.get(key);
    }
    
    // Create new box element
    const box = document.createElement('div');
    box.className = `detection-box tree-box ${category}-box`;
    
    // Add to pool and to container
    this._objectPool.boxes.set(key, box);
    this.boxesContainer.appendChild(box);
    
    return box;
  }
  
  /**
   * Get a segmentation mask from the object pool or create a new one
   * @private
   * @param {string|number} id - Unique identifier for the mask
   * @returns {HTMLCanvasElement} - The canvas element
   */
  _getSegmentationMaskFromPool(id) {
    // Check if element exists in pool
    if (this._objectPool.masks.has(id)) {
      return this._objectPool.masks.get(id);
    }
    
    // Create new canvas element
    const canvas = document.createElement('canvas');
    canvas.className = 'segmentation-mask';
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.pointerEvents = 'none';
    canvas.style.opacity = '0.6';
    
    // Add to pool
    this._objectPool.masks.set(id, canvas);
    
    return canvas;
  }
  
  /**
   * Get a label element from the object pool or create a new one
   * @private
   * @param {string|number} id - Unique identifier for the label
   * @returns {HTMLElement} - The label element
   */
  _getLabelFromPool(id) {
    // Check if element exists in pool
    if (this._objectPool.labels.has(id)) {
      return this._objectPool.labels.get(id);
    }
    
    // Create new label element
    const label = document.createElement('div');
    label.className = 'detection-label';
    label.style.position = 'absolute';
    label.style.top = '-20px';
    label.style.left = '0';
    label.style.padding = '2px 6px';
    label.style.fontSize = '11px';
    label.style.fontWeight = 'bold';
    label.style.borderRadius = '4px';
    label.style.whiteSpace = 'nowrap';
    label.style.zIndex = '11';
    
    // Add to pool
    this._objectPool.labels.set(id, label);
    
    return label;
  }
  
  /**
   * Update an existing canvas with new segmentation mask data
   * @private
   * @param {HTMLCanvasElement} canvas - Canvas element to update
   * @param {Object} segmentation - Segmentation data
   * @param {string} color - Color to use for the mask
   * @param {number} width - Width for the canvas
   * @param {number} height - Height for the canvas
   */
  _updateSegmentationMask(canvas, segmentation, color, width, height) {
    try {
      // Skip if missing required data
      if (!segmentation || !segmentation.size || (!segmentation.counts && !segmentation.data)) {
        return;
      }
      
      // Ensure canvas is visible
      canvas.style.display = 'block';
      
      // Update canvas size if needed
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
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
      
      // Get the drawing context
      const ctx = canvas.getContext('2d');
      
      // Clear previous content
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
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
      
      if (!mask) return;
      
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
      
    } catch (error) {
      console.error('Error updating segmentation mask:', error);
    }
  }
  
  /**
   * Called when the overlay is removed from the map
   */
  onRemove() {
    // Clean up event listeners
    if (this.map) {
      google.maps.event.clearListeners(this.map, 'idle');
      google.maps.event.clearListeners(this.map, 'zoom_changed');
      google.maps.event.clearListeners(this.map, 'dragend');
      google.maps.event.clearListeners(this.map, 'dragstart');
      google.maps.event.clearListeners(this.map, 'drag');
    }
    
    // Clear any pending draws
    if (this._drawTimeoutId) {
      clearTimeout(this._drawTimeoutId);
      this._drawTimeoutId = null;
    }
    
    // Clean up object pool
    if (this._objectPool) {
      this._objectPool.boxes.clear();
      this._objectPool.masks.clear();
      this._objectPool.labels.clear();
      this._objectPool = null;
    }
    
    // Remove DOM elements
    if (this.div && this.div.parentNode) {
      this.div.parentNode.removeChild(this.div);
      this.div = null;
      this.boxesContainer = null;
    }
  }
  
  /**
   * Update positions only during drag operations for better performance
   * This method only updates positions without re-creating elements
   * @private
   */
  _updatePositionsOnly() {
    // Skip if no projection or object pool
    const projection = this.getProjection();
    if (!projection || !this._objectPool || !this._objectPool.boxes.size) return;
    
    // Use requestAnimationFrame for smoother updates
    if (this._positionUpdateAnimFrame) {
      cancelAnimationFrame(this._positionUpdateAnimFrame);
    }
    
    this._positionUpdateAnimFrame = requestAnimationFrame(() => {
      const mapBounds = this.map.getBounds();
      if (!mapBounds) return;
      
      // Only update visible boxes
      this._objectPool.boxes.forEach((box, key) => {
        // Skip hidden boxes
        if (box.style.display === 'none') return;
        
        // Parse lat/lng from data attributes
        const lat = parseFloat(box.dataset.lat);
        const lng = parseFloat(box.dataset.lng);
        
        // Skip if invalid coordinates
        if (isNaN(lat) || isNaN(lng)) return;
        
        // Check if in current viewport (with larger buffer)
        const point = new google.maps.LatLng(lat, lng);
        if (!mapBounds.contains(point)) {
          // Hide if outside viewport
          box.style.display = 'none';
          return;
        }
        
        // Update position
        const pixelPoint = projection.fromLatLngToDivPixel(point);
        const width = parseFloat(box.style.width);
        const height = parseFloat(box.style.height);
        
        // Update position only
        box.style.left = `${pixelPoint.x - width/2}px`;
        box.style.top = `${pixelPoint.y - height/2}px`;
      });
    });
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
   * This method is kept empty to remove the counter functionality
   */
  addCountIndicator() {
    // Counter has been removed as requested
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
    
    // Indicator removed as requested
    this.div.appendChild(tintLayer);
  }
  
  /**
   * Add job ID indicator to the overlay
   * This method is kept empty to remove the job ID indicator
   */
  addJobIdIndicator(jobId) {
    // Job ID indicator has been removed as requested
  }
  
  /**
   * Update the overlay opacity
   */
  updateOpacity(opacity) {
    if (this.div) {
      this.div.style.opacity = opacity;
    }
    this.options.opacity = opacity;
  }
  
  /**
   * Update the class visibility
   */
  updateClasses(classes) {
    this.options.classes = classes;
    this.draw(); // Redraw with updated class visibility
  }
};

  console.log('MLDetectionOverlay class defined successfully');
  return true;
}

// Function to create and add a new overlay
function renderMLOverlay(map, data, options = {}) {
  // Ensure MLDetectionOverlay class is initialized first
  if (!ensureInitialized()) {
    console.error('Cannot render ML overlay - Google Maps API not fully loaded yet');
    
    // Set up a retry mechanism
    const retryDelay = 500; // ms
    const maxRetries = 6;   // 3 seconds total
    let retryCount = 0;
    
    const retryRender = () => {
      if (retryCount < maxRetries) {
        retryCount++;
        console.log(`Retry ${retryCount}/${maxRetries} to initialize MLOverlay`);
        
        if (ensureInitialized()) {
          console.log('MLOverlay successfully initialized on retry, rendering now');
          // Call this function again now that we're initialized
          return renderMLOverlay(map, data, options);
        } else {
          setTimeout(retryRender, retryDelay);
        }
      } else {
        console.error('Failed to initialize MLOverlay after maximum retries');
      }
    };
    
    // Start retry process
    setTimeout(retryRender, retryDelay);
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
    tintColor: options.tintColor || 'rgba(0, 30, 60, 0.3)',
    forceRenderBoxes: options.forceRenderBoxes !== undefined ? options.forceRenderBoxes : false
  };
  
  // Check for global visibility setting - CRITICAL for proper toggling
  // If overlay is explicitly disabled, respect that setting
  if (window.detectionShowOverlay === false && !opts.forceRenderBoxes) {
    console.log('MLOverlay: Overlay rendering skipped - visibility disabled');
    
    // If there's an existing overlay, remove it
    if (window._mlDetectionOverlay) {
      window._mlDetectionOverlay.setMap(null);
      window._mlDetectionOverlay = null;
    }
    
    return null;
  }
  
  // Verify the map is valid and has required methods
  if (!map || typeof map.getBounds !== 'function') {
    console.error('Invalid map object provided to renderMLOverlay');
    return null;
  }
  
  // Start rendering time measurement
  const startRenderTime = performance.now();
  
  try {
    // Determine if we need to preprocess the data
    let processedData;
    
    // If we already have a reference to the overlay and we're just updating settings,
    // we can reuse the existing processed data to avoid expensive reprocessing
    if (window._mlDetectionOverlay && options.updateOnly && !options.appendMode) {
      console.log('MLOverlay: Reusing existing data for faster refresh');
      processedData = window._mlDetectionOverlay.data;
    } else {
      // Preprocess the data to ensure it's in the correct format
      processedData = preprocessDetectionData(data);
    }
    
    // Check if the overlay exists but needs updating
    if (window._mlDetectionOverlay && !opts.appendMode) {
      console.log('MLOverlay: Updating existing overlay with new settings');
      
      // Update settings on the existing overlay
      if (options.opacity !== undefined) {
        window._mlDetectionOverlay.updateOpacity(options.opacity);
      }
      
      if (options.classes !== undefined) {
        window._mlDetectionOverlay.updateClasses(options.classes);
      }
      
      // Update the data if we have new data
      if (data && !options.updateOnly) {
        window._mlDetectionOverlay.data = processedData;
        window._mlDetectionOverlay.draw();
      }
      
      return window._mlDetectionOverlay;
    }
    
    // Remove existing overlay if not in append mode
    if (!opts.appendMode && window._mlDetectionOverlay) {
      window._mlDetectionOverlay.setMap(null);
      window._mlDetectionOverlay = null;
    }
    
    // Create a new overlay with processed data
    console.log(`MLOverlay: Creating new overlay with ${processedData.trees?.length || 0} trees`);
    const overlay = new MLDetectionOverlay(map, processedData, opts);
    
    // Store reference globally for later access
    window._mlDetectionOverlay = overlay;
    
    // Update global settings for consistency with other components
    window.mlOverlaySettings = window.mlOverlaySettings || {};
    window.mlOverlaySettings.showOverlay = true;
    window.mlOverlaySettings.showSegmentation = opts.showSegmentation;
    window.mlOverlaySettings.opacity = opts.opacity;
    window.mlOverlaySettings.showCategories = opts.classes;
    
    // Log rendering performance
    const renderTime = performance.now() - startRenderTime;
    console.log(`MLOverlayView: Overlay added to map (${renderTime.toFixed(2)}ms)`);
    
    return overlay;
  } catch (error) {
    console.error('MLOverlayView: Failed to create overlay', error);
    return null;
  }
}

/**
 * Preprocess detection data to ensure it's in the correct format for MLDetectionOverlay
 * Optimized version with performance improvements for large datasets
 */
function preprocessDetectionData(data) {
  if (!data) return { trees: [] };
  
  // Start performance timer
  const startTime = performance.now();
  
  // Avoid deep copy for large datasets - instead, create a new minimal object
  // This significantly reduces memory usage and processing time
  const processedData = {
    trees: [],
    buildings: [],
    power_lines: [],
    job_id: data.job_id,
    timestamp: data.timestamp,
    metadata: data.metadata
  };
  
  // Process metrics for debugging
  const metrics = {
    initialObjectCount: 0,
    convertedObjects: 0,
    processedObjects: 0,
    validatedObjects: 0,
    processingTime: 0
  };
  
  // If we have a detections array but no trees array, convert detections to trees
  if (Array.isArray(data.detections)) {
    metrics.initialObjectCount = data.detections.length;
    console.log(`MLOverlay: Processing ${data.detections.length} detections for efficient rendering`);
    
    // Use batch processing for large datasets
    const BATCH_SIZE = 200;
    
    for (let i = 0; i < data.detections.length; i += BATCH_SIZE) {
      const batch = data.detections.slice(i, i + BATCH_SIZE);
      
      batch.forEach(detection => {
        // Create a lightweight tree object with only necessary properties
        const tree = {
          id: detection.id || `tree-${i}-${Math.floor(Math.random() * 10000)}`,
          confidence: detection.confidence || 0.5,
          class: detection.class || 'tree',
          category: detection.category
        };
        
        // Process box coordinates efficiently
        if (detection.box) {
          // Already has a box object, validate and use it directly
          tree.box = {
            x: detection.box.x,
            y: detection.box.y,
            width: detection.box.width,
            height: detection.box.height
          };
        } else if (detection.bbox && Array.isArray(detection.bbox) && detection.bbox.length === 4) {
          // Convert bbox array to box object
          tree.box = {
            x: detection.bbox[0],
            y: detection.bbox[1],
            width: detection.bbox[2] - detection.bbox[0],
            height: detection.bbox[3] - detection.bbox[1]
          };
        } else if (detection.detection && detection.detection.bbox && Array.isArray(detection.detection.bbox)) {
          // Extract from nested detection property
          const bbox = detection.detection.bbox;
          tree.box = {
            x: bbox[0],
            y: bbox[1],
            width: bbox[2] - bbox[0],
            height: bbox[3] - bbox[1]
          };
        } else {
          // Default box at center if no valid coordinates
          tree.box = { x: 0.45, y: 0.45, width: 0.1, height: 0.1 };
        }
        
        // Only copy segmentation data if it exists and is needed
        // This saves memory for large datasets
        if (detection.segmentation) {
          tree.segmentation = detection.segmentation;
        }
        
        // Copy any additional properties needed for display
        if (detection.species) tree.species = detection.species;
        if (detection.risk_level) tree.risk_level = detection.risk_level;
        if (detection.height) tree.height = detection.height;
        if (detection.diameter) tree.diameter = detection.diameter;
        if (detection.address) tree.address = detection.address;
        if (detection.location) tree.location = detection.location;
        
        // Categorize by class dynamically
        const normalizedClass = (detection.class || 'tree').toLowerCase();
        
        if (normalizedClass.includes('tree') || normalizedClass.includes('vegetation')) {
          processedData.trees.push(tree);
        } else if (normalizedClass.includes('building') || normalizedClass.includes('roof')) {
          processedData.buildings.push(tree);
        } else if (normalizedClass.includes('power') || normalizedClass.includes('line')) {
          processedData.power_lines.push(tree);
        } else {
          // Default to trees for unknown classes
          processedData.trees.push(tree);
        }
        
        metrics.processedObjects++;
      });
    }
  } 
  // If we already have categorized arrays, process them efficiently
  else {
    // Process trees array if it exists
    if (Array.isArray(data.trees)) {
      metrics.initialObjectCount += data.trees.length;
      
      // Process in batches for large datasets
      const BATCH_SIZE = 200;
      
      for (let i = 0; i < data.trees.length; i += BATCH_SIZE) {
        const batch = data.trees.slice(i, i + BATCH_SIZE);
        
        batch.forEach(origTree => {
          // Create lightweight copy with only necessary properties
          const tree = { 
            id: origTree.id || `tree-${i}-${Math.floor(Math.random() * 10000)}`,
            confidence: origTree.confidence || 0.5,
            class: origTree.class || 'tree',
            category: origTree.category
          };
          
          // Efficiently handle box coordinates
          if (origTree.box && typeof origTree.box === 'object' && 
              typeof origTree.box.width === 'number' && 
              typeof origTree.box.height === 'number') {
            tree.box = {
              x: origTree.box.x,
              y: origTree.box.y,
              width: origTree.box.width,
              height: origTree.box.height
            };
          } else if (origTree.bbox && Array.isArray(origTree.bbox) && origTree.bbox.length === 4) {
            tree.box = {
              x: origTree.bbox[0],
              y: origTree.bbox[1],
              width: origTree.bbox[2] - origTree.bbox[0],
              height: origTree.bbox[3] - origTree.bbox[1]
            };
          } else if (origTree.detection && origTree.detection.bbox && Array.isArray(origTree.detection.bbox)) {
            const bbox = origTree.detection.bbox;
            tree.box = {
              x: bbox[0],
              y: bbox[1],
              width: bbox[2] - bbox[0],
              height: bbox[3] - bbox[1]
            };
          } else {
            // Default box at center if no valid coordinates
            tree.box = { x: 0.45, y: 0.45, width: 0.1, height: 0.1 };
          }
          
          // Only copy segmentation data if needed
          if (origTree.segmentation) {
            tree.segmentation = origTree.segmentation;
          }
          
          // Copy additional display properties
          if (origTree.species) tree.species = origTree.species;
          if (origTree.risk_level) tree.risk_level = origTree.risk_level;
          if (origTree.height) tree.height = origTree.height;
          if (origTree.diameter) tree.diameter = origTree.diameter;
          if (origTree.address) tree.address = origTree.address;
          if (origTree.location) tree.location = origTree.location;
          
          // Add to the processed trees array
          processedData.trees.push(tree);
          metrics.processedObjects++;
        });
      }
    }
    
    // Process buildings in the same way
    if (Array.isArray(data.buildings)) {
      metrics.initialObjectCount += data.buildings.length;
      processedData.buildings = data.buildings.map(building => ({
        ...building,
        box: building.box || {
          x: building.bbox ? building.bbox[0] : 0.45,
          y: building.bbox ? building.bbox[1] : 0.45,
          width: building.bbox ? (building.bbox[2] - building.bbox[0]) : 0.1,
          height: building.bbox ? (building.bbox[3] - building.bbox[1]) : 0.1
        }
      }));
      metrics.processedObjects += data.buildings.length;
    }
    
    // Process power lines in the same way
    if (Array.isArray(data.power_lines)) {
      metrics.initialObjectCount += data.power_lines.length;
      processedData.power_lines = data.power_lines.map(line => ({
        ...line,
        box: line.box || {
          x: line.bbox ? line.bbox[0] : 0.45,
          y: line.bbox ? line.bbox[1] : 0.45,
          width: line.bbox ? (line.bbox[2] - line.bbox[0]) : 0.1,
          height: line.bbox ? (line.bbox[3] - line.bbox[1]) : 0.1
        }
      }));
      metrics.processedObjects += data.power_lines.length;
    }
  }
  
  // Calculate and log performance metrics
  metrics.processingTime = performance.now() - startTime;
  console.log('MLOverlay data preprocessing metrics:', {
    initialCount: metrics.initialObjectCount,
    processedObjects: metrics.processedObjects,
    trees: processedData.trees.length,
    buildings: processedData.buildings.length,
    powerLines: processedData.power_lines.length,
    processingTime: `${metrics.processingTime.toFixed(2)}ms`,
    objectsPerSecond: `${Math.round((1000 * metrics.processedObjects) / metrics.processingTime)}/s`
  });
  
  return processedData;
}

// Function to remove existing overlay
function removeMLOverlay() {
  if (window._mlDetectionOverlay) {
    window._mlDetectionOverlay.setMap(null);
    window._mlDetectionOverlay = null;
    return true;
  }
  return false;
}

// Function to update opacity
function updateMLOverlayOpacity(opacity) {
  if (!opacity && opacity !== 0) {
    console.error('MLOverlay: Invalid opacity value:', opacity);
    return false;
  }
  
  console.log(`MLOverlay: Attempting to update overlay opacity to ${opacity}`);
  
  // Ensure opacity is a valid number between 0 and 1
  const validOpacity = Math.min(1, Math.max(0, parseFloat(opacity)));
  
  // Save to localStorage for persistence first to ensure settings are preserved
  try {
    localStorage.setItem('ml-overlay-opacity', validOpacity.toString());
    console.log(`MLOverlay: Saved opacity ${validOpacity} to localStorage`);
  } catch (e) {
    console.error("Error saving opacity to localStorage:", e);
  }
  
  // Update global settings for components to use
  if (window.mlOverlaySettings) {
    window.mlOverlaySettings.opacity = validOpacity;
  } else {
    window.mlOverlaySettings = { opacity: validOpacity };
  }
  
  let updated = false;
  
  // Try multiple ways to update the overlay
  
  // 1. Update through object method (most reliable if object exists)
  if (window._mlDetectionOverlay) {
    // Update overlay via the object method first
    window._mlDetectionOverlay.updateOpacity(validOpacity);
    updated = true;
    
    // Also directly update the background color of the div for immediate feedback
    if (window._mlDetectionOverlay.div) {
      window._mlDetectionOverlay.div.style.backgroundColor = `rgba(0, 30, 60, ${validOpacity})`;
      console.log(`MLOverlay: Updated div background color with opacity ${validOpacity}`);
      
      // Update the opacity of the container itself too
      window._mlDetectionOverlay.div.style.opacity = "1"; // Keep main div fully opaque
      
      // Update box container opacity if it exists
      const boxContainer = window._mlDetectionOverlay.div.querySelector('#detection-boxes-container');
      if (boxContainer) {
        boxContainer.style.opacity = validOpacity.toString();
      }
    }
  }
  
  // 2. Try direct DOM element access (fallback)
  const overlayEl = document.getElementById('ml-detection-overlay');
  if (overlayEl) {
    // Update the background color - this is what gives the tinted effect
    overlayEl.style.backgroundColor = `rgba(0, 30, 60, ${validOpacity})`;
    // Keep the element fully visible but adjust the background opacity
    overlayEl.style.opacity = "1";
    
    // Also update any box container within the overlay
    const boxContainer = overlayEl.querySelector('#detection-boxes-container');
    if (boxContainer) {
      boxContainer.style.opacity = validOpacity.toString();
    }
    
    console.log(`MLOverlay: Updated element background color with opacity ${validOpacity}`);
    updated = true;
  }
  
  // 3. Look for any detection boxes and update them directly
  const allBoxes = document.querySelectorAll('.detection-box');
  if (allBoxes.length > 0) {
    console.log(`MLOverlay: Updating ${allBoxes.length} detection boxes to opacity ${validOpacity}`);
    allBoxes.forEach(box => {
      box.style.opacity = validOpacity.toString();
    });
    updated = true;
  }
  
  // 4. Update any segmentation masks
  const allMasks = document.querySelectorAll('.segmentation-mask');
  if (allMasks.length > 0) {
    console.log(`MLOverlay: Updating ${allMasks.length} segmentation masks to opacity ${validOpacity}`);
    allMasks.forEach(mask => {
      mask.style.opacity = validOpacity.toString();
    });
    updated = true;
  }
  
  // Dispatch event to notify other components about the change
  window.dispatchEvent(new CustomEvent('mlOverlayOpacityChanged', { 
    detail: { opacity: validOpacity } 
  }));
  
  return updated;
}

// Function to update class visibility
function updateMLOverlayClasses(classes) {
  if (window._mlDetectionOverlay) {
    window._mlDetectionOverlay.updateClasses(classes);
    return true;
  }
  return false;
}

// Create backward-compatible functions to match the old interface
function renderDetectionOverlay(detectionData, options = {}) {
  try {
    // Check if Google Maps API is loaded
    if (typeof google === 'undefined' || !google.maps) {
      console.warn('MLOverlay: Google Maps API not fully loaded yet, will retry when ready');
      
      // Set up a listener for the Maps API initialization if not already done
      if (!window._detectionOverlayMapWaiter) {
        window._detectionOverlayMapWaiter = true;
        
        console.log('Setting up Google Maps initialization listener for detection overlay');
        
        // Set up a MutationObserver to check for Google Maps initialization
        const checkGoogleMaps = () => {
          if (typeof google !== 'undefined' && google.maps) {
            console.log('Google Maps detected via interval, rendering detection overlay');
            renderDetectionOverlay(detectionData, options);
            return true;
          }
          return false;
        };
        
        // Try immediately
        if (checkGoogleMaps()) return true;
        
        // If Maps API isn't loaded yet, set up an interval to check periodically
        const mapCheckInterval = setInterval(() => {
          if (checkGoogleMaps()) {
            clearInterval(mapCheckInterval);
          }
        }, 500);
        
        // Also listen for the custom event from useGoogleMapsApi hook
        window.addEventListener('mapsApiInitialized', () => {
          console.log('Maps API initialized event received, rendering detection overlay');
          clearInterval(mapCheckInterval);
          renderDetectionOverlay(detectionData, options);
        }, { once: true });
      }
      
      return false;
    }
    
    // Get map instance with better fallbacks
    let mapInstance = null;
    
    // Try all the possible map references we might have
    if (window.map && typeof window.map.getBounds === 'function') {
      console.log('MLOverlay: Using window.map reference');
      mapInstance = window.map;
    } else if (window.googleMapsInstance && typeof window.googleMapsInstance.getBounds === 'function') {
      console.log('MLOverlay: Using window.googleMapsInstance reference');
      mapInstance = window.googleMapsInstance;
    } else if (window._googleMap && typeof window._googleMap.getBounds === 'function') {
      console.log('MLOverlay: Using window._googleMap reference');
      mapInstance = window._googleMap;
    } else if (window.mapRef && window.mapRef.current && typeof window.mapRef.current.getBounds === 'function') {
      console.log('MLOverlay: Using window.mapRef.current reference');
      mapInstance = window.mapRef.current;
    } else if (window._mapRef && window._mapRef.current && typeof window._mapRef.current.getBounds === 'function') {
      console.log('MLOverlay: Using window._mapRef.current reference');
      mapInstance = window._mapRef.current;
    }
    
    if (!mapInstance) {
      console.error('MLOverlay: Could not find a valid Google Maps instance');
      return false;
    }
    
    // Ensure proper target element
    const targetElement = options.targetElement || 
                         document.getElementById('map-container') || 
                         document.getElementById('map');
    
    if (!targetElement) {
      console.error('MLOverlay: Map container not found');
      return false;
    }
    
    // Use the new implementation
    const overlay = renderMLOverlay(mapInstance, detectionData, options);
    return !!overlay;
  } catch (error) {
    console.error('MLOverlay: Error in renderDetectionOverlay:', error);
    return false;
  }
}

function removeDetectionOverlay() {
  return removeMLOverlay();
}

function updateOverlayOpacity(opacity) {
  return updateMLOverlayOpacity(opacity);
}

function updateDetectionClasses(classes) {
  return updateMLOverlayClasses(classes);
}

// Export both new and backward-compatible functions
export { 
  MLDetectionOverlay,
  renderMLOverlay,
  removeMLOverlay,
  updateMLOverlayOpacity,
  updateMLOverlayClasses,
  // Backward compatibility exports
  renderDetectionOverlay,
  removeDetectionOverlay,
  updateOverlayOpacity,
  updateDetectionClasses
};

// Initialize MLOverlay class when the module loads
initMLOverlayClass();

// Create wrapper for class initialization on demand
const ensureInitialized = () => {
  if (!MLDetectionOverlay) {
    const initialized = initMLOverlayClass();
    if (!initialized) {
      console.error('Google Maps API not fully loaded yet, MLOverlay functions may not work');
      
      // Try to set up a listener for Maps API loading if not yet initialized
      if (!window._mlOverlayInitAttempted && window.addEventListener) {
        window._mlOverlayInitAttempted = true;
        
        console.log('Setting up Google Maps initialization listener for MLOverlay');
        
        // Listen for Google Maps initialization event
        window.addEventListener('mapsApiInitialized', () => {
          console.log('Maps API initialized event received, initializing MLOverlay classes');
          initMLOverlayClass();
          
          // Also expose to global scope when Maps API becomes available
          addToGlobalScope();
        });
      }
      
      return false;
    }
  }
  
  return true;
};

// Function to add all classes and methods to global scope
const addToGlobalScope = () => {
  // First ensure class is initialized
  if (!ensureInitialized()) return;
  
  // Add to global scope for script interop
  window.MLDetectionOverlay = MLDetectionOverlay;
  window.renderMLOverlay = renderMLOverlay;
  window.removeMLOverlay = removeMLOverlay;
  window.updateMLOverlayOpacity = updateMLOverlayOpacity;
  window.updateMLOverlayClasses = updateMLOverlayClasses;

  // Add backward compatibility globals
  window.renderDetectionOverlay = renderDetectionOverlay;
  window.removeDetectionOverlay = removeDetectionOverlay;
  window.updateOverlayOpacity = updateOverlayOpacity;
  window.updateDetectionClasses = updateDetectionClasses;
  
  console.log('ML Overlay functions added to global scope');
};

// Try to add to global scope immediately
addToGlobalScope();

// Also set up a window load event listener as a backup
window.addEventListener('load', () => {
  console.log('Window load event - checking MLOverlay initialization');
  addToGlobalScope();
  
  // Double check after a short delay to ensure Maps API has fully loaded
  setTimeout(addToGlobalScope, 1000);
});

// Export default for module imports
export default {
  renderDetectionOverlay,
  removeDetectionOverlay,
  updateOverlayOpacity,
  updateDetectionClasses,
  MLDetectionOverlay,
  ensureInitialized
};