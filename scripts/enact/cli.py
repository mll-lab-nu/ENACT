#!/usr/bin/env python3
"""
ENACT CLI - Unified command-line interface

Main entry point providing subcommands for all ENACT functionality:
  - segment: Process task directories for frame segmentation
  - qa: Generate QA pairs from segmented trajectories
  - eval: Evaluate model outputs on ordering tasks
  - download: Download datasets and resources
"""

import sys
import argparse
from pathlib import Path


def segment_command(args):
    """Run segmentation processor."""
    from enact.processors.segmentation_processor import SegmentationProcessor
    
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
    
    processor = SegmentationProcessor(args.input_root, args.output_root)
    processor.process_all_tasks()


def qa_command(args):
    """Run QA generation processor."""
    from enact.processors.qa_gen_processor import BatchQAProcessor
    
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
    
    processor = BatchQAProcessor(
        args.input_root, 
        args.output_file, 
        args.raw_data_dir, 
        seed=args.seed, 
        num_to_sample=args.num_to_sample
    )
    processor.run()


def eval_command(args):
    """Run evaluation processor."""
    from enact.processors.evaluator_processor import EvaluatorProcessor
    
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
    
    processor = EvaluatorProcessor(
        input_path=args.input_path,
        segmented_data_dir=args.segmented_data,
        raw_data_dir=args.raw_data,
        output_root=args.output_root,
        analyze_wrong_cases=args.analyze_wrong_cases
    )
    processor.process_all_files()


def download_command(args):
    """Download datasets and resources."""
    print(f"Downloading: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    
    if args.dataset == "sample":
        print("\nDownloading sample dataset...")
        print("⚠️  Dataset download functionality not yet implemented.")
        print("Please manually download datasets from the ENACT repository.")
    elif args.dataset == "full":
        print("\nDownloading full dataset...")
        print("⚠️  Dataset download functionality not yet implemented.")
        print("Please manually download datasets from the ENACT repository.")
    else:
        print(f"Unknown dataset: {args.dataset}")
        print("Available datasets: sample, full")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='enact',
        description='ENACT - Frame segmentation, QA generation, and evaluation framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Segmentation
  enact segment data/replayed_activities data/segmented_activities
  enact segment --dry-run
  
  # QA Generation
  enact qa data/segmented_activities data/replayed_activities data/QA/enact_ordering.jsonl
  enact qa --seed 42 --num-to-sample 10
  
  # Evaluation
  enact eval model_outputs/ --analyze-wrong-cases
  enact eval model_output.jsonl --dry-run
  
  # Download datasets
  enact download sample --output-dir data/
  enact download full --output-dir data/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Segment subcommand
    segment_parser = subparsers.add_parser(
        'segment',
        help='Process task directories for frame segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    segment_parser.add_argument(
        'input_root',
        nargs='?',
        default='data/replayed_activities',
        help='Path to task directory or root directory containing task directories'
    )
    segment_parser.add_argument(
        'output_root',
        nargs='?',
        default='data/segmented_activities',
        help='Root directory for output structure'
    )
    segment_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually doing it'
    )
    segment_parser.set_defaults(func=segment_command)
    
    # QA subcommand
    qa_parser = subparsers.add_parser(
        'qa',
        help='Generate QA pairs from segmented trajectories',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    qa_parser.add_argument(
        'input_root',
        nargs='?',
        default='data/segmented_activities',
        help='Root directory containing segmented task directories'
    )
    qa_parser.add_argument(
        'raw_data_dir',
        nargs='?',
        default='data/replayed_activities',
        help='Root directory containing raw data'
    )
    qa_parser.add_argument(
        'output_file',
        nargs='?',
        default='data/QA/enact_ordering.jsonl',
        help='Output JSONL file path'
    )
    qa_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually generating QA pairs'
    )
    qa_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible results'
    )
    qa_parser.add_argument(
        '--num-to-sample',
        type=int,
        default=5,
        help='Number of sequences to sample for multi-step generators'
    )
    qa_parser.set_defaults(func=qa_command)
    
    # Eval subcommand
    eval_parser = subparsers.add_parser(
        'eval',
        help='Evaluate model outputs on ordering tasks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    eval_parser.add_argument(
        'input_path',
        help='Path to JSONL file or directory containing JSONL files'
    )
    eval_parser.add_argument(
        '--segmented-data',
        default='data/segmented_activities',
        help='Path to segmented activities data directory'
    )
    eval_parser.add_argument(
        '--raw-data',
        default='data/replayed_activities',
        help='Path to raw replayed activities data directory'
    )
    eval_parser.add_argument(
        '--output-root',
        default='data/evaluation',
        help='Root directory for evaluation output'
    )
    eval_parser.add_argument(
        '--analyze-wrong-cases',
        action='store_true',
        help='Enable wrong case analysis and save signatures'
    )
    eval_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually doing it'
    )
    eval_parser.set_defaults(func=eval_command)
    
    # Download subcommand
    download_parser = subparsers.add_parser(
        'download',
        help='Download datasets and resources',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    download_parser.add_argument(
        'dataset',
        choices=['sample', 'full'],
        help='Dataset to download'
    )
    download_parser.add_argument(
        '--output-dir',
        default='data/',
        help='Output directory for downloaded data'
    )
    download_parser.set_defaults(func=download_command)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


