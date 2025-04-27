"""
Analytics API Routes
------------------
Endpoints for data analytics, BigQuery integration, and Gemini AI analysis.
"""

from flask import Blueprint, jsonify, request, current_app
import asyncio
import json
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
analytics_bp = Blueprint('analytics', __name__, url_prefix='/api/analytics')

# Route for BigQuery analytics queries
@analytics_bp.route('/query/<query_type>', methods=['GET'])
async def run_analytics_query(query_type):
    """Run a predefined analytics query"""
    try:
        # Validate if query type is valid
        valid_query_types = [
            'risk_distribution', 
            'species_distribution', 
            'geographical_hotspots',
            'temporal_analysis'
        ]
        
        if query_type not in valid_query_types:
            return jsonify({
                "success": False,
                "message": f"Invalid query type. Valid options are: {', '.join(valid_query_types)}"
            }), 400
        
        # Get parameters from query string
        params = {}
        for key in request.args:
            params[key] = request.args.get(key)
        
        # Get BigQuery service from app context
        bigquery_service = current_app.bigquery_service
        
        # Run query
        result = await bigquery_service.run_analytics_query(query_type, params)
        
        if result.get('success', False):
            return jsonify(result)
        else:
            return jsonify(result), 500
    except Exception as e:
        logger.error(f"Error in analytics query: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Analytics query error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

# Route for tree risk analysis with Gemini
@analytics_bp.route('/tree/<tree_id>/analyze', methods=['GET'])
async def analyze_tree(tree_id):
    """Analyze a tree using Gemini AI"""
    try:
        # Get services from app context
        db_service = current_app.db_service
        gemini_service = current_app.gemini_service
        
        # Get tree data from database
        tree_data = await db_service.get_tree(tree_id)
        
        if not tree_data:
            return jsonify({
                "success": False,
                "message": f"Tree with ID {tree_id} not found",
                "timestamp": datetime.now().isoformat()
            }), 404
        
        # Run analysis with Gemini
        analysis_result = await gemini_service.analyze_tree_risk(tree_data)
        
        if analysis_result.get('success', False):
            return jsonify(analysis_result)
        else:
            return jsonify(analysis_result), 500
    except Exception as e:
        logger.error(f"Error analyzing tree: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Tree analysis error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

# Route for generating a property report
@analytics_bp.route('/property/<property_id>/report', methods=['POST'])
async def generate_property_report(property_id):
    """Generate a comprehensive property report"""
    try:
        # Get services from app context
        db_service = current_app.db_service
        gemini_service = current_app.gemini_service
        
        # Get property data
        property_data = await db_service.get_property(property_id)
        
        if not property_data:
            return jsonify({
                "success": False,
                "message": f"Property with ID {property_id} not found",
                "timestamp": datetime.now().isoformat()
            }), 404
        
        # Get trees for this property
        trees_data = await db_service.get_trees_by_property(property_id)
        
        # Generate report with Gemini
        report_result = await gemini_service.generate_property_report(property_data, trees_data)
        
        if report_result.get('success', False):
            return jsonify(report_result)
        else:
            return jsonify(report_result), 500
    except Exception as e:
        logger.error(f"Error generating property report: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Report generation error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

# Route for syncing data to BigQuery
@analytics_bp.route('/sync/bigquery', methods=['POST'])
async def sync_to_bigquery():
    """Sync local data to BigQuery for analytics"""
    try:
        # Get request data
        data = request.json or {}
        entity_type = data.get('entity_type', 'trees')
        limit = data.get('limit', 1000)
        
        # Get services from app context
        db_service = current_app.db_service
        bigquery_service = current_app.bigquery_service
        
        if entity_type == 'trees':
            # Get tree data
            all_properties = await db_service.get_properties()
            all_trees = []
            
            # Get trees for each property
            for prop in all_properties[:limit]:
                property_id = prop.get('id')
                property_trees = await db_service.get_trees_by_property(property_id)
                all_trees.extend(property_trees)
            
            # Sync to BigQuery
            result = await bigquery_service.sync_trees_to_bigquery(all_trees)
            
            if result.get('success', False):
                return jsonify(result)
            else:
                return jsonify(result), 500
        else:
            return jsonify({
                "success": False,
                "message": f"Unsupported entity type: {entity_type}. Supported: trees",
                "timestamp": datetime.now().isoformat()
            }), 400
    except Exception as e:
        logger.error(f"Error syncing to BigQuery: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Sync error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

# Route for tree detection using Gemini AI
@analytics_bp.route('/gemini/detect-trees', methods=['POST'])
async def gemini_detect_trees():
    """Detect trees using Gemini AI from map data
    
    This endpoint provides a direct way for the frontend to use Gemini AI for tree detection
    without going through the ML pipeline. It uses the map view information directly from
    the Google Maps JavaScript API to generate accurate tree locations using Gemini's 
    natural language capabilities."""
    try:
        # Get services from app context
        gemini_service = current_app.gemini_service
        
        # Get request data
        data = request.json or {}
        map_view_info = data.get('map_view_info', {})
        job_id = data.get('job_id', f"gemini_{int(datetime.now().timestamp())}")
        
        if not map_view_info:
            return jsonify({
                "success": False,
                "message": "Map view information is required",
                "timestamp": datetime.now().isoformat()
            }), 400
            
        # Process with Gemini - ensure service is initialized
        if not gemini_service.is_initialized:
            logger.info("Initializing Gemini service before detection")
            await gemini_service.initialize()
                
        # Call the detection with full logging
        logger.info(f"Calling Gemini detection with map_view_info: {map_view_info}")
        detection_result = await gemini_service.detect_trees_from_map_data(
            map_view_info, 
            job_id
        )
            
        # Log the result
        tree_count = detection_result.get('tree_count', 0)
        logger.info(f"Gemini detection complete with {tree_count} trees detected")
        
        return jsonify(detection_result)
        
    except Exception as e:
        logger.error(f"Error detecting trees with Gemini: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Detection error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

# Route for creating BigQuery view
@analytics_bp.route('/bigquery/create-view', methods=['POST'])
async def create_bigquery_view():
    """Create a materialized view in BigQuery"""
    try:
        # Get BigQuery service from app context
        bigquery_service = current_app.bigquery_service
        
        # Create the view
        result = await bigquery_service.create_risk_view()
        
        if result.get('success', False):
            return jsonify(result)
        else:
            return jsonify(result), 500
    except Exception as e:
        logger.error(f"Error creating BigQuery view: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"View creation error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

# Route for getting map data with analytics
@analytics_bp.route('/map/data', methods=['GET'])
async def get_map_analytics_data():
    """Get map data with analytics for visualization"""
    try:
        # Get bounds parameters
        bounds = {
            'north': float(request.args.get('north', 90)),
            'south': float(request.args.get('south', -90)),
            'east': float(request.args.get('east', 180)),
            'west': float(request.args.get('west', -180))
        }
        
        # Get filters
        filters = {}
        for key in ['risk_level', 'species']:
            if key in request.args:
                filters[key] = request.args.get(key)
        
        # Get database service from app context
        db_service = current_app.db_service
        
        # Get geospatial data
        result = await db_service.get_geospatial_data(bounds, filters)
        
        if result.get('success', False):
            return jsonify(result)
        else:
            return jsonify(result), 500
    except Exception as e:
        logger.error(f"Error fetching map analytics data: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Map data error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500