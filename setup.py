#!/usr/bin/env python3
"""
Setup script for ENACT project
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="enact",
    version="0.1.0",
    description="ENACT: Frame segmentation, QA generation, and evaluation framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ENACT Team",
    python_requires=">=3.10",
    packages=find_packages(include=["enact", "enact.*"]),
    install_requires=[
        "numpy>=1.20.0",
        "Pillow>=8.0.0",
        "tqdm>=4.60.0",
        "opencv-python>=4.5.0",
    ],
    entry_points={
        'console_scripts': [
            'enact-segment=scripts.enact.run_segmentation:main',
            'enact-qa=scripts.enact.run_qa_generation:main',
            'enact-eval=scripts.enact.run_eval:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
    ],
)

