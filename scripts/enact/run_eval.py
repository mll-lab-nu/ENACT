#!/usr/bin/env python3
"""
Evaluation Script

Entry point for evaluating model outputs on ordering tasks.
Supports both single file mode and batch mode.
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from enact.processors.evaluator_processor import EvaluatorProcessor
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Evaluate model outputs on ordering tasks (supports single file or batch mode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'input_path',
        nargs='?',
        default='/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/Fixed_data/evaluation/new_model_outputs',
        help='Path to JSONL file (single mode) or directory containing JSONL files (batch mode)'
    )
    
    parser.add_argument(
        '--segmented-data',
        default='data/segmented_activities',
        help='Path to segmented activities data directory'
    )
    
    parser.add_argument(
        '--raw-data',
        default='data/replayed_activities',
        help='Path to raw replayed activities data directory'
    )
    
    parser.add_argument(
        '--output-root',
        default='data/evaluation',
        help='Root directory for evaluation output (contains meta_performance and detailed_eval subdirectories)'
    )
    
    parser.add_argument(
        '--analyze-wrong-cases',
        action='store_true',
        help='Enable wrong case analysis and save signatures to signatures/ folder'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually doing it'
    )
    
    args = parser.parse_args()
    
    print(f"Input path: {args.input_path}")
    print(f"Segmented data: {args.segmented_data}")
    print(f"Raw data: {args.raw_data}")
    print(f"Output root: {args.output_root}")
    
    if args.dry_run:
        print("\n[DRY RUN MODE] - No files will be processed")
        input_path = Path(args.input_path)
        
        if not input_path.exists():
            print(f"Input path does not exist: {args.input_path}")
            return
        
        if input_path.is_file():
            print(f"Single file mode: {input_path.name}")
            print(f"Would evaluate: {input_path}")
        else:
            import glob
            pattern = str(input_path / "enact_ordering_*.jsonl")
            jsonl_files = glob.glob(pattern)
            print(f"Batch mode: Would evaluate {len(jsonl_files)} files:")
            for file in sorted(jsonl_files):
                print(f"  - {Path(file).name}")
        
        print(f"\nResults would be saved to:")
        print(f"  - Meta performance: {args.output_root}/meta_performance/")
        print(f"  - Detailed evaluations: {args.output_root}/detailed_eval/")
        if args.analyze_wrong_cases:
            print(f"  - Wrong case signatures: {args.output_root}/signatures/")
        return
    
    try:
        processor = EvaluatorProcessor(
            input_path=args.input_path,
            segmented_data_dir=args.segmented_data,
            raw_data_dir=args.raw_data,
            output_root=args.output_root,
            analyze_wrong_cases=args.analyze_wrong_cases
        )
        processor.process_all_files()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

