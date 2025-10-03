#!/usr/bin/env python3
"""
Batch QA Generation Script

Entry point for batch question-answer pair generation from segmented trajectories.
"""

import sys
import argparse
from pathlib import Path

try:
    from enact.processors.qa_gen_processor import BatchQAProcessor
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Batch generate QA pairs from segmented trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'input_root',
        nargs='?',
        default='data/segmented_activities',
        help='Root directory containing segmented task directories'
    )
    
    parser.add_argument(
        'raw_data_dir',
        nargs='?',
        default='data/replayed_activities',
        help='Root directory containing raw data'
    )
    
    parser.add_argument(
        'output_file',
        nargs='?',
        default='data/QA/enact_ordering.jsonl',
        help='Output JSONL file path'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually generating QA pairs'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible results'
    )
    
    parser.add_argument(
        '--num-to-sample',
        type=int,
        default=5,
        help='Number of sequences to sample for multi-step generators'
    )
    
    args = parser.parse_args()
    
    print(f"Input root: {args.input_root}")
    print(f"Raw data dir: {args.raw_data_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Seed: {args.seed}")
    print(f"Num to sample: {args.num_to_sample}")
    
    if args.dry_run:
        print("\n[DRY RUN MODE] - No QA pairs will be generated")
        input_root = Path(args.input_root)
        if input_root.exists():
            task_dirs = [d for d in input_root.iterdir() 
                        if d.is_dir() and (d / "segmented_scene_graph_0.json").exists()]
            print(f"Would process {len(task_dirs)} task directories:")
            for task_dir in sorted(task_dirs):
                print(f"  - {task_dir.name}")
        else:
            print(f"Input directory does not exist: {args.input_root}")
        return
    
    try:
        processor = BatchQAProcessor(
            args.input_root, 
            args.output_file, 
            args.raw_data_dir, 
            seed=args.seed, 
            num_to_sample=args.num_to_sample
        )
        processor.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

