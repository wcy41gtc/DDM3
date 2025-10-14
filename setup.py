#!/usr/bin/env python
"""Setup script for DDM3D package."""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ddm3d",
    version="0.3.0",
    author="DDM3D Contributors",
    author_email="your.email@example.com",
    description="3D Displacement Discontinuity Method for DAS Simulation with Comprehensive Examples and Dynamic Interpolation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/wcy41gtc/DDM3",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Geology",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0,<2.0.0",
        "matplotlib>=3.3.0,<4.0.0",
        "h5py>=3.0.0,<4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0,<8.0.0",
            "pytest-cov>=2.0.0,<5.0.0",
            "black>=21.0.0,<24.0.0",
            "flake8>=3.8.0,<7.0.0",
            "mypy>=0.800,<2.0.0",
        ],
        "optional": [
            "scipy>=1.7.0,<2.0.0",
            "pandas>=1.3.0,<3.0.0",
            "seaborn>=0.11.0,<1.0.0",
        ],
        "all": [
            "pytest>=6.0.0,<8.0.0",
            "pytest-cov>=2.0.0,<5.0.0",
            "black>=21.0.0,<24.0.0",
            "flake8>=3.8.0,<7.0.0",
            "mypy>=0.800,<2.0.0",
            "scipy>=1.7.0,<2.0.0",
            "pandas>=1.3.0,<3.0.0",
            "seaborn>=0.11.0,<1.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
