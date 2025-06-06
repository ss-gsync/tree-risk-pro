import json
import os
import logging
from datetime import datetime # Assuming datetime is used elsewhere for logging or timestamps
# Assuming other necessary imports like io, base64, PIL are available if this were standalone

logger = logging.getLogger(__name__) # Standard practice for logging

# Placeholder for S2Manager if it's a custom class
class S2ManagerPlaceholder:
    def get_cell_ids_for_tree(self, lat, lng):
        # Replace with actual S2 cell ID generation logic
        logger.debug(f"Generating S2 cell IDs for lat: {lat}, lng: {lng}")
        return [f"s2_cell_for_{lat}_{lng}"]

class YourClass: # Assuming these methods are part of a class
    def __init__(self):
        # Initialize any necessary attributes, e.g., temp_dir, s2_manager
        self.temp_dir = "/tmp/ttt_data" # Example
        os.makedirs(self.temp_dir, exist_ok=True)
        self.s2_manager = S2ManagerPlaceholder() # Use the placeholder or your actual S2Manager
        # self.ml_service = ... (if used by other methods)
        # self.detect_trees = ... (if used by other methods)

    # --- Other methods like _get_pixel_to_latlon_mapping, _normalize_image_coords_to_geo would be here ---

    def _get_pixel_to_latlon_mapping(self, image_path, bounds):
        """
        Placeholder: Calculates the mapping between pixel coordinates and lat/lon.
        This is CRITICAL for accurate geo-referencing.
        For an orthorectified image (like a satellite image from Static Maps API),
        this would involve knowing the image dimensions and the precise geographic
        coordinates of its corners (from the 'bounds' input).

        Args:
            image_path (str): Path to the image.
            bounds (list): Geographic bounds [[sw_lng, sw_lat], [ne_lng, ne_lat]].

        Returns:
            dict: Mapping information including image_width, image_height,
                  and parameters needed for coordinate transformation (e.g., geotransform).
        """
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            logger.error(f"Could not open image {image_path} to get dimensions: {e}")
            # Fallback dimensions, but this will lead to inaccurate geo-referencing
            img_width, img_height = 640, 640 # Default or raise error

        # sw_lng, sw_lat = bounds[0][0], bounds[0][1]
        # ne_lng, ne_lat = bounds[1][0], bounds[1][1]

        # This is a simplified example. Real georeferencing is more complex.
        # For a simple north-up image, you can calculate pixel resolution:
        # lng_per_pixel = (ne_lng - sw_lng) / img_width
        # lat_per_pixel = (ne_lat - sw_lat) / img_height # Typically negative if origin is top-left

        # The 'geotransform' array is commonly used by GDAL:
        # GeoTransform[0] /* top left x */
        # GeoTransform[1] /* w-e pixel resolution */
        # GeoTransform[2] /* 0 if image is "north up" */
        # GeoTransform[3] /* top left y */
        # GeoTransform[4] /* 0 if image is "north up" */
        # GeoTransform[5] /* n-s pixel resolution (negative value) */
        # For simplicity, we'll pass bounds and dimensions, and _normalize_image_coords_to_geo will use them.
        return {
            'image_width': img_width,
            'image_height': img_height,
            'bounds': bounds, # Pass bounds through for the conversion function
            # 'geotransform': [sw_lng, lng_per_pixel, 0, ne_lat, 0, -lat_per_pixel] # Example geotransform
        }

    def _normalize_image_coords_to_geo(self, norm_x, norm_y, mapping_info):
        """
        Placeholder: Converts normalized image coordinates (0-1) to geographic lat/lon.
        
        Args:
            norm_x (float): Normalized x-coordinate (0-1, from left).
            norm_y (float): Normalized y-coordinate (0-1, from top).
            mapping_info (dict): Output from _get_pixel_to_latlon_mapping.
                                 Expected to contain 'bounds', 'image_width', 'image_height'.
        
        Returns:
            tuple: (longitude, latitude)
        """
        bounds = mapping_info['bounds']
        # img_width = mapping_info['image_width'] # Not directly needed if using normalized
        # img_height = mapping_info['image_height'] # Not directly needed if using normalized

        sw_lng, sw_lat = bounds[0][0], bounds[0][1]
        ne_lng, ne_lat = bounds[1][0], bounds[1][1]

        # Linear interpolation based on normalized coordinates
        geo_lng = sw_lng + norm_x * (ne_lng - sw_lng)
        # For latitude, if origin is top-left, norm_y=0 is ne_lat, norm_y=1 is sw_lat
        geo_lat = ne_lat - norm_y * (ne_lat - sw_lat) # Corrected interpolation for latitude

        return geo_lng, geo_lat

    async def _detect_trees_with_gemini(self, image_path, bounds, job_id, ml_response_dir):
        """
        Use Gemini API to detect trees in an image, expecting normalized coordinates.

        Args:
            image_path (str): Path to the satellite image.
            bounds (list): Geographic bounds [[sw_lng, sw_lat], [ne_lng, ne_lat]] of the image_path.
            job_id (str): Job ID for tracking.
            ml_response_dir (str): Directory to store response.

        Returns:
            dict: Detection results.
        """
        try:
            logger.info(f"Starting tree detection with Gemini for job {job_id}, image: {image_path}")

            # Dynamically import GeminiService to avoid circular dependencies or allow optional install
            try:
                from .gemini_service import GeminiService # Assuming it's in the same directory or discoverable
            except ImportError:
                logger.error("GeminiService could not be imported. Ensure it's available.")
                return {"error": "GeminiService not available", "job_id": job_id, "trees": []}

            gemini_service = GeminiService()
            # Assuming async initialization if needed, or remove await if synchronous
            # await gemini_service.initialize() # If your GeminiService has an async init

            # The prompt sent within gemini_service.analyze_image_for_trees should ask for:
            # - A JSON list of tree objects.
            # - Each tree object to contain:
            #   - "normalized_bbox": [ymin, xmin, ymax, xmax] (coordinates 0-1 relative to image dimensions)
            #   - "height_meters": estimated height in meters
            #   - "species": best guess of tree species or "Unknown"
            #   - "confidence": detection confidence (0-1)
            #   - "notes" (optional): any relevant notes from Gemini
            # Example prompt for GeminiService to use:
            # "Please analyze this satellite image to detect all trees. For each tree, provide its
            #  normalized bounding box as [ymin, xmin, ymax, xmax] where coordinates are between 0.0 and 1.0.
            #  Also provide the estimated height in meters, the likely species (or 'Unknown'),
            #  and your confidence score for the detection.
            #  Respond with a JSON list of tree objects, where each object has keys:
            #  'normalized_bbox', 'height_meters', 'species', 'confidence', and 'notes'."

            gemini_api_result = await gemini_service.analyze_image_for_trees(
                image_path=image_path,
                # Pass bounds or other context if GeminiService needs it for its internal prompt
                # context_info={"geographic_bounds": bounds}
            )

            if not gemini_api_result or not gemini_api_result.get('success', False):
                error_msg = gemini_api_result.get('message', 'Gemini analysis failed or returned empty.')
                logger.error(f"Gemini analysis failed for job {job_id}: {error_msg}")
                return {"error": error_msg, "job_id": job_id, "trees": [], "ml_response_dir": ml_response_dir}

            # Assuming gemini_api_result['response'] is the direct JSON list from Gemini
            # or a string that needs to be parsed into that list.
            raw_gemini_output = gemini_api_result.get('response')
            detected_trees_from_gemini = []

            if isinstance(raw_gemini_output, list): # If GeminiService already parsed it to a list
                detected_trees_from_gemini = raw_gemini_output
            elif isinstance(raw_gemini_output, str):
                try:
                    # Clean up the response if it's wrapped in markdown code blocks
                    if raw_gemini_output.strip().startswith("```json"):
                        raw_gemini_output = raw_gemini_output.strip()[7:]
                        if raw_gemini_output.strip().endswith("```"):
                             raw_gemini_output = raw_gemini_output.strip()[:-3]
                    
                    detected_trees_from_gemini = json.loads(raw_gemini_output)
                    if not isinstance(detected_trees_from_gemini, list):
                        logger.warning(f"Gemini returned JSON but not a list for job {job_id}. Output: {str(raw_gemini_output)[:200]}")
                        detected_trees_from_gemini = [] # Reset if not a list
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing Gemini JSON response for job {job_id}: {e}. Response: {raw_gemini_output[:500]}")
                    return {"error": f"Invalid JSON from Gemini: {e}", "job_id": job_id, "trees": [], "ml_response_dir": ml_response_dir}
            else:
                logger.warning(f"Unexpected Gemini response type for job {job_id}: {type(raw_gemini_output)}")


            processed_trees_data = []
            # This mapping is crucial and depends on the image being orthorectified
            # and 'bounds' accurately representing the image's geographic extent.
            geo_mapping_info = self._get_pixel_to_latlon_mapping(image_path, bounds)

            for i, tree_info_from_gemini in enumerate(detected_trees_from_gemini):
                norm_bbox = tree_info_from_gemini.get('normalized_bbox')
                if not (isinstance(norm_bbox, list) and len(norm_bbox) == 4 and all(isinstance(c, (int, float)) for c in norm_bbox)):
                    logger.warning(f"Skipping tree due to invalid normalized_bbox: {norm_bbox} for job {job_id}")
                    continue
                
                # Ensure coordinates are within 0-1, clip if necessary
                ymin, xmin, ymax, xmax = [max(0.0, min(1.0, c)) for c in norm_bbox]

                # Calculate center of the normalized bounding box
                center_norm_x = (xmin + xmax) / 2
                center_norm_y = (ymin + ymax) / 2

                try:
                    geo_lng, geo_lat = self._normalize_image_coords_to_geo(center_norm_x, center_norm_y, geo_mapping_info)
                except Exception as e:
                    logger.error(f"Error converting normalized coords to geo for tree {i} in job {job_id}: {e}")
                    continue # Skip this tree if geocoding fails

                s2_cell_ids = self.s2_manager.get_cell_ids_for_tree(geo_lat, geo_lng)

                tree_entry = {
                    'id': tree_info_from_gemini.get('id', f"{job_id}_gemini_tree_{i+1}"),
                    # Store the normalized bbox as Gemini provided it (image-space)
                    'normalized_bbox_image': [ymin, xmin, ymax, xmax],
                    'confidence': tree_info_from_gemini.get('confidence', 0.75), # Default confidence
                    'location': [geo_lng, geo_lat], # Geo-coordinates
                    'height': tree_info_from_gemini.get('height_meters', 10.0), # Default height
                    'species': tree_info_from_gemini.get('species', 'Unknown'),
                    'notes': tree_info_from_gemini.get('notes', ''),
                    'detection_type': 'gemini',
                    's2_cells': s2_cell_ids,
                    # Optionally store original image dimensions for context
                    'image_width_px': geo_mapping_info['image_width'],
                    'image_height_px': geo_mapping_info['image_height']
                }
                processed_trees_data.append(tree_entry)
            
            logger.info(f"Successfully processed {len(processed_trees_data)} trees from Gemini for job {job_id}")

            # --- Save results (similar to your original code) ---
            # Save the raw Gemini API response for debugging
            raw_gemini_response_path = os.path.join(ml_response_dir, f'gemini_raw_output_{job_id}.json')
            with open(raw_gemini_response_path, 'w') as f:
                # Save the original gemini_api_result which might contain more than just the response list
                json.dump(gemini_api_result, f, indent=2)

            # Create zarr-like data format (or your desired final format)
            # For simplicity, I'm adapting your zarr_data structure slightly
            output_data_for_client = {
                "job_id": job_id,
                "status": "complete",
                "mode": "detection_gemini", # Indicate mode
                "frames": [{
                    "frame_id": 0, # Assuming single image
                    "image_path": image_path, # Relative or absolute path to image used
                    "image_width_px": geo_mapping_info['image_width'],
                    'image_height_px': geo_mapping_info['image_height'],
                    "geographic_bounds": bounds,
                    "tree_count": len(processed_trees_data),
                    "trees": processed_trees_data,
                }],
                "summary": gemini_api_result.get('summary', f"Detected {len(processed_trees_data)} trees using Gemini."),
                "ml_response_dir": ml_response_dir
            }

            trees_json_path = os.path.join(ml_response_dir, 'trees_gemini.json')
            with open(trees_json_path, 'w') as f:
                json.dump(output_data_for_client, f, indent=2)
            logger.info(f"Saved Gemini detection results to {trees_json_path}")

            return {
                "job_id": job_id,
                "tree_count": len(processed_trees_data),
                "trees": processed_trees_data, # Return the list of processed tree objects
                "summary": output_data_for_client["summary"],
                "ml_response_dir": ml_response_dir,
                "status": "complete"
            }

        except Exception as e:
            logger.error(f"Unhandled error in _detect_trees_with_gemini for job {job_id}: {str(e)}", exc_info=True)
            return {"error": str(e), "job_id": job_id, "trees": [], "ml_response_dir": ml_response_dir or "not_created"}

    # --- detect_trees_from_map_view (with comments for geo-referencing context) ---
    async def detect_trees_from_map_view(self, map_view_info, job_id):
        """
        Run tree detection based on map view information, primarily using a custom ML pipeline.
        This function sets up the image and then calls a specific ML backend.
        Geo-referencing of results from this pipeline depends on accurate 'bounds' for the
        image processed by the custom ML.
        """
        mode = 'segmentation' # Default mode, might be overridden by custom ML
        ml_response_dir = None
        image_path = None # Initialize image_path

        # ... (initial logging and view_data extraction as in your original code) ...
        # Log entry point with clear markers for tracing
        logger.info(f"Starting ML detection from map view - Job ID: {job_id}")
        
        # Extract view data
        view_data = map_view_info.get('viewData', {})
        
        # Force ML pipeline settings (as per your original logic for this function)
        map_view_info['use_gemini'] = False
        map_view_info['use_deepforest'] = True # Assuming this directs to your custom pipeline
        map_view_info['segmentation_mode'] = True # Or determined by 'mode'
        map_view_info['use_in_memory_service'] = True

        # Create a SINGLE directory for all ML results using the job ID
        timestamp_from_job_id = job_id.replace('detection_', '') if job_id.startswith('detection_') else job_id
        ml_dir_name = f"ml_{timestamp_from_job_id}"
        base_results_dir = os.path.join(self.temp_dir, ml_dir_name) # self.temp_dir needs to be defined
        os.makedirs(base_results_dir, exist_ok=True)
        ml_response_dir = base_results_dir # Use this as the primary output directory
        logger.info(f"Using ML results directory: {ml_response_dir}")

        # ... (canvas capture logic from your original code - saving the image to image_path) ...
        # This part is crucial: 'image_path' must be set correctly.
        # 'bounds' must accurately reflect the geographic extent of 'image_path'.

        has_canvas_capture = False
        # [Your existing extensive canvas capture logic here, ensuring image_path is set
        # and saved into ml_response_dir if successfully captured/decoded]
        # For brevity, I'm omitting the detailed canvas capture block but it should:
        # 1. Try to get imageUrl from view_data.
        # 2. Decode and save it to `image_path = os.path.join(ml_response_dir, f'capture_{timestamp_from_job_id}.jpg')`
        # 3. Set `has_canvas_capture = True`
        # Example placeholder for where canvas capture image saving would happen:
        canvas_image_url = view_data.get('imageUrl')
        if canvas_image_url and (canvas_image_url.startswith('data:image/jpeg;base64,') or canvas_image_url.startswith('data:image/png;base64,')):
            try:
                # Simplified saving logic from your code
                b64_data = canvas_image_url.split(',', 1)[1]
                image_bytes = base64.b64decode(b64_data)
                # from PIL import Image # Ensure imported
                # import io # Ensure imported
                # img_pil = Image.open(io.BytesIO(image_bytes)) # Verify
                
                image_path = os.path.join(ml_response_dir, f'capture_{timestamp_from_job_id}.jpg')
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                logger.info(f"Saved canvas capture to {image_path}")
                has_canvas_capture = True
                # CRITICAL FOR GEOREFERENCING: The 'bounds' from map_view_info.get('viewData', {}).get('bounds')
                # are assumed to correspond to this canvas capture.
            except Exception as e:
                logger.error(f"Error processing canvas capture: {e}", exc_info=True)
                has_canvas_capture = False
        # ... (rest of your canvas capture checking logic) ...

        # Fallback to Google Maps Static API if no canvas capture
        if not has_canvas_capture and not map_view_info.get('skip_canvas_capture', False):
            logger.info(f"Job {job_id}: No canvas capture, attempting Google Maps Static API.")
            try:
                # ... (Your logic for calculating center/zoom if needed from view_data['bounds']) ...
                # Make sure view_data['bounds'] is present and valid if used here.

                # Define the path for the static map image within the job's directory
                gmaps_image_filename = f'gmaps_satellite_{timestamp_from_job_id}.jpg'
                gmaps_image_path_for_ml = os.path.join(ml_response_dir, gmaps_image_filename)

                # Call _get_satellite_image_from_gmaps
                # This method needs to ensure it returns:
                # 1. success (bool)
                # 2. image_data_dict (containing 'imageUrl', 'mapWidth', 'mapHeight')
                # 3. IMPORTANT FOR GEOREF: The *actual geographic bounds* of the fetched static image.
                #    The Static Maps API returns an image of a certain pixel size for a given center/zoom.
                #    Its precise geographic corners need to be calculated or known.
                #    Let's assume _get_satellite_image_from_gmaps saves the image to gmaps_image_path_for_ml
                #    and updates view_data['bounds'] with the *actual* bounds of this fetched image.
                
                # success, gmaps_img_dict = await self._get_satellite_image_from_gmaps(view_data, output_path=gmaps_image_path_for_ml)
                # For this example, let's mock this call slightly:
                # Assume _get_satellite_image_from_gmaps saves the image and returns its actual path and NEW bounds
                # Placeholder for the actual call, which should update view_data['bounds'] if successful
                # For now, if this path is taken, we'll assume the original view_data['bounds'] are still the best guess
                # unless _get_satellite_image_from_gmaps explicitly provides more accurate ones for the static image.
                
                # This is a conceptual call, your actual implementation of _get_satellite_image_from_gmaps is key
                # success_gmaps, updated_view_data_with_gmaps_image_and_actual_bounds = \
                #     await self._get_satellite_image_from_gmaps_and_actual_bounds(view_data, gmaps_image_path_for_ml)

                # if success_gmaps:
                #     image_path = gmaps_image_path_for_ml
                #     view_data.update(updated_view_data_with_gmaps_image_and_actual_bounds) # This should update bounds
                #     logger.info(f"Job {job_id}: Successfully used Google Maps Static API. Image: {image_path}")
                #     has_canvas_capture = True # Flag that we have an image
                # else:
                #     logger.warning(f"Job {job_id}: Google Maps Static API failed. No image for ML.")
                #     # ... (your dummy image fallback logic) ...
                #     # If dummy image is created, image_path should point to it.
                #     # Bounds for dummy image are essentially meaningless for geo-referencing real trees.
                logger.warning(f"Job {job_id}: Google Maps Static API image retrieval logic needs to be robustly implemented here.")
                # For now, if we reach here without an image_path, we must fail or use a dummy.
                # The dummy image creation logic from your original code would fit here.
                # If a dummy image is created, its image_path should be set.
                # However, geo-referencing detections from a dummy image is not meaningful.

            except Exception as e:
                logger.error(f"Job {job_id}: Error in Google Maps Static API retrieval: {e}", exc_info=True)
        
        if not image_path or not os.path.exists(image_path):
            logger.error(f"Job {job_id}: No valid image (canvas or static map) available for ML detection.")
            return {
                "error": "No image available for ML detection after attempting canvas and static map.",
                "job_id": job_id, "status": "error", "ml_response_dir": ml_response_dir
            }

        # CRITICAL: Ensure 'bounds_for_ml_image' accurately reflects the geographic extent of 'image_path'
        bounds_for_ml_image = view_data.get('bounds')
        if not bounds_for_ml_image: # This should have been set by canvas capture or Gmaps Static API image prep
            logger.error(f"Job {job_id}: Geographic bounds for the ML image '{image_path}' are missing!")
            return {"error": "Missing geographic bounds for ML image.", "job_id": job_id, "status": "error", "ml_response_dir": ml_response_dir}
        
        logger.info(f"Job {job_id}: Proceeding with ML detection on image: {image_path} with bounds: {bounds_for_ml_image}")

        # ... (Your existing ML service/DeepForest loading and prediction logic) ...
        # This part calls your custom (non-Gemini) ML pipeline.
        # Example:
        # result_from_custom_ml = await self.detect_trees_from_canvas_capture(
        # image_path=image_path,
        # bounds=bounds_for_ml_image, # Pass the correct bounds for this image
        # job_id=job_id,
        # ml_response_dir=ml_response_dir,
        # existing_trees=existing_trees # If applicable
        # )
        # The 'result_from_custom_ml' should contain tree detections with bounding boxes
        # relative to 'image_path' (either pixel or normalized).
        # If these results need to be geo-referenced later, you'll use 'image_path',
        # its dimensions, and 'bounds_for_ml_image'.
        
        # For the sake of this example, I'll simulate a result from your custom ML.
        # Your actual call to DeepForest or `detect_trees_from_canvas_capture` would go here.
        # This simulated result assumes pixel bounding boxes.
        simulated_custom_ml_trees = [
            {'bbox': [10, 20, 50, 60], 'confidence': 0.9, 'class_id': 0, 'detection_type': 'deepforest_simulated'},
            {'bbox': [100, 120, 150, 180], 'confidence': 0.85, 'class_id': 0, 'detection_type': 'deepforest_simulated'}
        ]
        result = {
            "job_id": job_id,
            "tree_count": len(simulated_custom_ml_trees),
            "trees": simulated_custom_ml_trees, # These are in image-space (pixel or normalized)
            "ml_response_dir": ml_response_dir,
            "image_path_used": image_path, # For clarity
            "image_bounds_used": bounds_for_ml_image, # For clarity
            "status": "complete",
            "mode": mode # segmentation or detection
        }
        # ... (Your existing segmentation logic _add_segmentation_data(ml_response_dir, result['trees'], image_path, bounds_for_ml_image))


        # IMPORTANT FOR GEO-REFERENCING:
        # The 'result['trees']' contains bounding boxes in the coordinate system of 'image_path'.
        # To get geo-locations, you would iterate through these trees, get their
        # pixel/normalized coordinates, and use a function similar to
        # _normalize_image_coords_to_geo (or a pixel-to-geo variant) along with
        # the 'bounds_for_ml_image' and dimensions of 'image_path'.

        logger.info(f"ML detection (custom pipeline) complete for job {job_id}: Found {result.get('tree_count', 0)} trees.")
        return result

        # ... (Your existing error handling) ...