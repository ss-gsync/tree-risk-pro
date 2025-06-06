"""
Database Service
---------------
Handles connections to the master PostgreSQL database and synchronizes data
between the local cache and cloud infrastructure.
"""

import os
import json
import logging
import asyncio
import asyncpg
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseService:
    """Service for database operations and cloud synchronization"""

    def __init__(self, config_path: str = None):
        """Initialize the database service"""
        self.config = self._load_config(config_path)
        self.pool = None
        self.is_connected = False
        self.sync_task = None

    async def initialize(self):
        """Initialize the database connection pool"""
        try:
            # Create connection pool to PostgreSQL
            self.pool = await asyncpg.create_pool(
                host=self.config['database']['host'],
                port=self.config['database']['port'],
                user=self.config['database']['user'],
                password=self.config['database']['password'],
                database=self.config['database']['dbname'],
                min_size=self.config['database']['min_connections'],
                max_size=self.config['database']['max_connections']
            )
            
            self.is_connected = True
            logger.info("Successfully connected to master database")
            
            # Start background sync task
            self.sync_task = asyncio.create_task(self._sync_background())
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to master database: {str(e)}")
            self.is_connected = False
            return False

    async def get_properties(self, filters: Dict = None) -> List[Dict]:
        """Get properties from the master database"""
        if not self.is_connected:
            logger.warning("Not connected to master database, using local cache")
            return await self._get_local_properties(filters)
        
        try:
            query = """
                SELECT p.*, 
                       COUNT(t.id) as tree_count, 
                       SUM(CASE WHEN t.risk_level = 'high' THEN 1 ELSE 0 END) as high_risk_count
                FROM properties p
                LEFT JOIN trees t ON p.id = t.property_id
            """
            
            where_clauses = []
            params = []
            
            if filters:
                if 'id' in filters:
                    where_clauses.append("p.id = $1")
                    params.append(filters['id'])
                
                if 'address' in filters:
                    where_clauses.append("p.address ILIKE $" + str(len(params) + 1))
                    params.append(f"%{filters['address']}%")
                
                if 'city' in filters:
                    where_clauses.append("p.city ILIKE $" + str(len(params) + 1))
                    params.append(f"%{filters['city']}%")
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            query += " GROUP BY p.id ORDER BY p.created_at DESC"
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                properties = [dict(row) for row in rows]
                
                # Cache the results locally
                await self._update_local_cache('properties', properties)
                
                return properties
        except Exception as e:
            logger.error(f"Error fetching properties from master database: {str(e)}")
            return await self._get_local_properties(filters)

    async def get_trees_by_property(self, property_id: str) -> List[Dict]:
        """Get trees for a specific property"""
        if not self.is_connected:
            logger.warning("Not connected to master database, using local cache")
            return await self._get_local_trees(property_id)
        
        try:
            query = """
                SELECT t.*, 
                       a.risk_factors,
                       a.updated_at as assessment_date
                FROM trees t
                LEFT JOIN assessments a ON t.id = a.tree_id
                WHERE t.property_id = $1
                ORDER BY t.created_at DESC
            """
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, property_id)
                trees = [dict(row) for row in rows]
                
                # Cache the results locally
                await self._update_local_cache(f'trees_{property_id}', trees)
                
                return trees
        except Exception as e:
            logger.error(f"Error fetching trees from master database: {str(e)}")
            return await self._get_local_trees(property_id)

    async def sync_data(self) -> Dict:
        """Manually trigger synchronization with master database"""
        if not self.is_connected:
            return {
                "success": False,
                "message": "Not connected to master database",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Get local data that needs to be synchronized
            properties_updated = await self._sync_properties()
            trees_updated = await self._sync_trees()
            assessments_updated = await self._sync_assessments()
            
            return {
                "success": True,
                "properties_updated": properties_updated,
                "trees_updated": trees_updated,
                "assessments_updated": assessments_updated,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error during data synchronization: {str(e)}")
            return {
                "success": False,
                "message": f"Synchronization error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def get_analytics(self, query_type: str, params: Dict = None) -> Dict:
        """Fetch analytics data from cloud infrastructure"""
        if not self.is_connected:
            return {
                "success": False,
                "message": "Not connected to analytics service",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            analytics_queries = {
                "risk_summary": """
                    SELECT risk_level, COUNT(*) as count 
                    FROM trees 
                    GROUP BY risk_level
                    ORDER BY CASE 
                        WHEN risk_level = 'high' THEN 1
                        WHEN risk_level = 'medium' THEN 2
                        WHEN risk_level = 'low' THEN 3
                        ELSE 4
                    END
                """,
                "property_stats": """
                    SELECT p.id, p.address, COUNT(t.id) as tree_count,
                           SUM(CASE WHEN t.risk_level = 'high' THEN 1 ELSE 0 END) as high_risk_count,
                           SUM(CASE WHEN t.risk_level = 'medium' THEN 1 ELSE 0 END) as medium_risk_count,
                           SUM(CASE WHEN t.risk_level = 'low' THEN 1 ELSE 0 END) as low_risk_count
                    FROM properties p
                    LEFT JOIN trees t ON p.id = t.property_id
                    GROUP BY p.id, p.address
                    ORDER BY high_risk_count DESC
                    LIMIT 10
                """,
                "temporal_analysis": """
                    SELECT date_trunc('month', a.created_at) as month,
                           t.risk_level,
                           COUNT(*) as count
                    FROM assessments a
                    JOIN trees t ON a.tree_id = t.id
                    WHERE a.created_at >= NOW() - INTERVAL '1 year'
                    GROUP BY month, t.risk_level
                    ORDER BY month, t.risk_level
                """
            }
            
            if query_type not in analytics_queries:
                return {
                    "success": False,
                    "message": f"Unknown analytics query type: {query_type}",
                    "timestamp": datetime.now().isoformat()
                }
            
            query = analytics_queries[query_type]
            
            # Apply filters if provided
            if params and query_type == "property_stats" and "city" in params:
                query = query.replace(
                    "FROM properties p", 
                    f"FROM properties p WHERE p.city ILIKE '%{params['city']}%'"
                )
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query)
                data = [dict(row) for row in rows]
                
                return {
                    "success": True,
                    "query_type": query_type,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error fetching analytics data: {str(e)}")
            return {
                "success": False,
                "message": f"Analytics query error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def get_geospatial_data(self, bounds: Dict, filters: Dict = None) -> Dict:
        """Fetch geospatial data within specified bounds"""
        if not self.is_connected:
            return {
                "success": False,
                "message": "Not connected to geospatial service",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Basic bounds validation
            if not all(key in bounds for key in ['north', 'south', 'east', 'west']):
                return {
                    "success": False,
                    "message": "Invalid bounds specified",
                    "timestamp": datetime.now().isoformat()
                }
            
            query = """
                SELECT t.id, t.property_id, t.species, t.risk_level,
                       t.height, t.canopy_width, t.latitude, t.longitude,
                       p.address
                FROM trees t
                JOIN properties p ON t.property_id = p.id
                WHERE t.latitude BETWEEN $1 AND $2
                AND t.longitude BETWEEN $3 AND $4
            """
            
            params = [bounds['south'], bounds['north'], bounds['west'], bounds['east']]
            param_index = 5
            
            # Apply additional filters if provided
            if filters:
                if 'risk_level' in filters:
                    query += f" AND t.risk_level = ${param_index}"
                    params.append(filters['risk_level'])
                    param_index += 1
                
                if 'species' in filters:
                    query += f" AND t.species ILIKE ${param_index}"
                    params.append(f"%{filters['species']}%")
                    param_index += 1
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                trees = [dict(row) for row in rows]
                
                return {
                    "success": True,
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [tree['longitude'], tree['latitude']]
                            },
                            "properties": {
                                **{k: v for k, v in tree.items() if k not in ['longitude', 'latitude']}
                            }
                        }
                        for tree in trees
                    ],
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error fetching geospatial data: {str(e)}")
            return {
                "success": False,
                "message": f"Geospatial query error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def close(self):
        """Close database connections and shutdown service"""
        if self.sync_task:
            self.sync_task.cancel()
            try:
                await self.sync_task
            except asyncio.CancelledError:
                pass
        
        if self.pool:
            await self.pool.close()
        
        self.is_connected = False
        logger.info("Database service has been shut down")

    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "database": {
                "host": os.environ.get("DB_HOST", "localhost"),
                "port": int(os.environ.get("DB_PORT", "5432")),
                "user": os.environ.get("DB_USER", "postgres"),
                "password": os.environ.get("DB_PASSWORD", "postgres"),
                "dbname": os.environ.get("DB_NAME", "tree_risk_assessment"),
                "min_connections": 5,
                "max_connections": 20
            },
            "sync": {
                "interval_seconds": 300,  # 5 minutes
                "local_cache_dir": os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    default_config.update(config)
            except Exception as e:
                logger.error(f"Error loading config file: {str(e)}")
        
        # Ensure cache directory exists
        os.makedirs(default_config["sync"]["local_cache_dir"], exist_ok=True)
        
        return default_config

    async def _sync_background(self):
        """Background task to periodically sync with master database"""
        while True:
            try:
                # Only sync if connected
                if self.is_connected:
                    await self._sync_properties()
                    await self._sync_trees()
                    await self._sync_assessments()
                    logger.info("Background sync completed successfully")
            except Exception as e:
                logger.error(f"Background sync error: {str(e)}")
            
            # Wait for next sync interval
            await asyncio.sleep(self.config["sync"]["interval_seconds"])

    async def _sync_properties(self) -> int:
        """Sync properties with master database"""
        # Logic to sync properties
        # This is a simplified implementation - a real sync would be more complex
        try:
            async with self.pool.acquire() as conn:
                # Get properties from master db
                rows = await conn.fetch("SELECT * FROM properties ORDER BY created_at DESC")
                properties = [dict(row) for row in rows]
                
                # Update local cache
                await self._update_local_cache('properties', properties)
                
                return len(properties)
        except Exception as e:
            logger.error(f"Error syncing properties: {str(e)}")
            return 0

    async def _sync_trees(self) -> int:
        """Sync trees with master database"""
        try:
            async with self.pool.acquire() as conn:
                # Get all property IDs
                property_ids = await conn.fetch("SELECT DISTINCT id FROM properties")
                
                total_trees = 0
                for prop_row in property_ids:
                    property_id = prop_row['id']
                    
                    # Get trees for this property
                    rows = await conn.fetch(
                        "SELECT * FROM trees WHERE property_id = $1 ORDER BY created_at DESC", 
                        property_id
                    )
                    trees = [dict(row) for row in rows]
                    
                    # Update local cache
                    await self._update_local_cache(f'trees_{property_id}', trees)
                    
                    total_trees += len(trees)
                
                return total_trees
        except Exception as e:
            logger.error(f"Error syncing trees: {str(e)}")
            return 0

    async def _sync_assessments(self) -> int:
        """Sync assessments with master database"""
        try:
            async with self.pool.acquire() as conn:
                # Get assessments from master db
                rows = await conn.fetch("""
                    SELECT a.*, t.property_id 
                    FROM assessments a
                    JOIN trees t ON a.tree_id = t.id
                    ORDER BY a.updated_at DESC
                """)
                assessments = [dict(row) for row in rows]
                
                # Update local cache
                await self._update_local_cache('assessments', assessments)
                
                return len(assessments)
        except Exception as e:
            logger.error(f"Error syncing assessments: {str(e)}")
            return 0

    async def _get_local_properties(self, filters: Dict = None) -> List[Dict]:
        """Get properties from local cache"""
        try:
            properties = await self._read_local_cache('properties')
            if not properties:
                return []
            
            # Apply filters if provided
            if filters:
                filtered = properties
                
                if 'id' in filters:
                    filtered = [p for p in filtered if p['id'] == filters['id']]
                
                if 'address' in filters:
                    address_term = filters['address'].lower()
                    filtered = [p for p in filtered if address_term in p.get('address', '').lower()]
                
                if 'city' in filters:
                    city_term = filters['city'].lower()
                    filtered = [p for p in filtered if city_term in p.get('city', '').lower()]
                
                return filtered
            
            return properties
        except Exception as e:
            logger.error(f"Error reading local properties: {str(e)}")
            return []

    async def _get_local_trees(self, property_id: str) -> List[Dict]:
        """Get trees from local cache"""
        try:
            return await self._read_local_cache(f'trees_{property_id}') or []
        except Exception as e:
            logger.error(f"Error reading local trees: {str(e)}")
            return []

    async def _update_local_cache(self, cache_key: str, data: List[Dict]) -> bool:
        """Update local cache file"""
        try:
            cache_file = os.path.join(self.config["sync"]["local_cache_dir"], f"{cache_key}.json")
            
            # Write atomically by using a temporary file
            temp_file = f"{cache_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Rename is atomic on most file systems
            os.replace(temp_file, cache_file)
            
            return True
        except Exception as e:
            logger.error(f"Error updating local cache {cache_key}: {str(e)}")
            return False

    async def _read_local_cache(self, cache_key: str) -> List[Dict]:
        """Read data from local cache file"""
        try:
            cache_file = os.path.join(self.config["sync"]["local_cache_dir"], f"{cache_key}.json")
            
            if not os.path.exists(cache_file):
                return []
            
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading local cache {cache_key}: {str(e)}")
            return []