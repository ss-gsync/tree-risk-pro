#!/usr/bin/env python3
"""
Run all tests for Tree Risk Pro dashboard
"""

import os
import sys
import asyncio
import subprocess
import time

def print_separator(text):
    """Print a separator with text"""
    separator = "=" * 80
    print(f"\n{separator}")
    print(f"  {text}")
    print(f"{separator}\n")

async def main():
    """Run all test scripts"""
    scripts = [
        "test_ml.py",     # Test ML pipeline detection
        "test_gemini.py"  # Test Gemini API integration
    ]
    
    print_separator("Starting Test Suite for Tree Risk Pro Dashboard")
    
    # Run each test script
    for script in scripts:
        script_path = os.path.join(os.path.dirname(__file__), script)
        
        if os.path.exists(script_path):
            print_separator(f"Running {script}")
            
            # Run the script as a subprocess
            try:
                process = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                # Stream output in real-time
                for line in process.stdout:
                    print(line, end='')
                
                # Wait for process to complete
                process.wait()
                
                # Check exit code
                if process.returncode == 0:
                    print(f"\n✅ {script} completed successfully")
                else:
                    print(f"\n❌ {script} failed with exit code {process.returncode}")
                
            except Exception as e:
                print(f"\n❌ Error running {script}: {str(e)}")
        else:
            print(f"❌ Test script not found: {script_path}")
    
    print_separator("Test Suite Completed")

if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())