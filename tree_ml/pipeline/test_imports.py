"""
Test imports for the model server
"""

import os
import sys

# Add the necessary paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grounded-sam"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grounded-sam/GroundingDINO"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grounded-sam/segment_anything"))

print("Testing imports...")

try:
    # Import GroundingDINO
    from groundingdino.config import SLConfig
    print("✓ Imported SLConfig from groundingdino.config")
    
    # Import Segment Anything
    from segment_anything import sam_model_registry
    print("✓ Imported sam_model_registry from segment_anything")
    
    print("All imports successful!")
except Exception as e:
    print(f"Error importing modules: {str(e)}")