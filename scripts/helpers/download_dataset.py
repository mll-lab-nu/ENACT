#!/usr/bin/env python3
"""
Dataset Download Helper Script

Example helper script for downloading datasets.
Place additional helper scripts in this directory.
"""

import os
import sys
import argparse

# After installing the package with `pip install -e .`,
# you can import from enact without path manipulation:
# from enact.utils import ...


def main():
    """Main function for downloading dataset."""
    parser = argparse.ArgumentParser(
        description="Download ENACT dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--output-dir',
        default='./data',
        help='Output directory for downloaded dataset'
    )
    
    parser.add_argument(
        '--dataset',
        default='all',
        help='Which dataset to download (all, train, test)'
    )
    
    args = parser.parse_args()
    
    print(f"Downloading dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    
    # TODO: Implement your dataset download logic here
    print("TODO: Add download implementation")


if __name__ == "__main__":
    main()

