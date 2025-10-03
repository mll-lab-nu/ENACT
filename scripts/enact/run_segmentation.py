#!/usr/bin/env python3
"""
Segmentation Script

Entry point for processing task directories for frame segmentation.
Supports both single task mode and batch mode.
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from enact.processors.segmentation_processor import SegmentationProcessor
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Process task directories for frame segmentation (supports single task or batch mode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'input_root',
        nargs='?',
        default='data/replayed_activities',
        help='Path to task directory (single mode) or root directory containing task directories (batch mode)'
    )
    
    parser.add_argument(
        'output_root', 
        nargs='?',
        default='data/segmented_activities',
        help='Root directory for output structure'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually doing it'
    )
    
    args = parser.parse_args()
    
    print(f"Input root: {args.input_root}")
    print(f"Output root: {args.output_root}")
    
    if args.dry_run:
        print("\n[DRY RUN MODE] - No files will be processed")
        input_root = Path(args.input_root)
        if input_root.exists():
            task_dirs = [d for d in input_root.iterdir() 
                        if d.is_dir() and (d / "scene_graph_0.json").exists()]
            print(f"Would process {len(task_dirs)} task directories:")
            for task_dir in sorted(task_dirs):
                print(f"  - {task_dir.name}")
        else:
            print(f"Input directory does not exist: {args.input_root}")
        return
    
    try:
        processor = SegmentationProcessor(args.input_root, args.output_root)
        processor.process_all_tasks()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()