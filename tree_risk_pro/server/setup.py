from setuptools import setup, find_packages

setup(
    name="trp-server",
    version="0.1.0",
    description="Edge Cache Server for drone data with h5serv integration",
    author="G-Sync Team",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "aiohttp>=3.8.0",
        "fastapi>=0.95.0",
        "h5py>=3.8.0",
        "numpy>=1.24.0",
        "opencv-python>=4.7.0",
        "tqdm>=4.65.0",
        "ssl",
        "asyncio",
        "pathlib",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "pylint>=2.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "edge-cache-server=server:run_server",
        ],
    },
)