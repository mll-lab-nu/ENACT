#!/usr/bin/env python3
"""
Evaluator Processor

Processes model output files for evaluation.
Supports both single file mode (input is a specific JSONL file) and 
batch mode (input is a directory containing multiple JSONL files).

Uses the OrderingEvaluator class to perform evaluation and generate
detailed results and meta performance reports.
"""

import sys
import json
import glob
from pathlib import Path
from typing import Optional

try:
    from enact.core.evaluators import OrderingEvaluator
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class EvaluatorProcessor:
    """
    Processes model output files for evaluation.
    Supports both single file and batch processing modes.
    """
    
    def __init__(
        self, 
        input_path: str,
        segmented_data_dir: str = "data/segmented_activities",
        raw_data_dir: str = "data/replayed_activities",
        output_root: str = "data/evaluation",
        analyze_wrong_cases: bool = False
    ):
        """
        Initialize the evaluator processor.
        
        Args:
            input_path: For batch mode: directory containing multiple JSONL files.
                       For single mode: path to a specific JSONL file.
            segmented_data_dir: Directory containing segmented scene graphs
            raw_data_dir: Directory containing raw data and scene graphs
            output_root: Root directory for evaluation output (contains meta_performance and detailed_eval)
            analyze_wrong_cases: Whether to analyze and save wrong case signatures
        """
        self.input_path = Path(input_path)
        self.segmented_data_dir = Path(segmented_data_dir)
        self.raw_data_dir = Path(raw_data_dir)
        self.output_root = Path(output_root)
        self.analyze_wrong_cases = analyze_wrong_cases
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")
        
        if not self.segmented_data_dir.exists():
            raise FileNotFoundError(f"Segmented data directory not found: {segmented_data_dir}")
        
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")
        
        # Create output directories
        self.meta_performance_dir = self.output_root / "meta_performance"
        self.detailed_eval_dir = self.output_root / "detailed_eval"
        self.signatures_dir = self.output_root / "signatures"
        self.meta_performance_dir.mkdir(parents=True, exist_ok=True)
        self.detailed_eval_dir.mkdir(parents=True, exist_ok=True)
        if self.analyze_wrong_cases:
            self.signatures_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine processing mode based on input structure
        self.is_single_mode = self.input_path.is_file()
        
        if self.is_single_mode:
            print(f"Single file mode detected: {self.input_path.name}")
        else:
            print(f"Batch mode detected: processing multiple files")
        
        # Initialize the evaluator
        print("Initializing OrderingEvaluator...")
        self.evaluator = OrderingEvaluator(
            input_root_dir=str(self.segmented_data_dir),
            raw_data_dir=str(self.raw_data_dir)
        )
    
    def _extract_model_name_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract model name from filename like 'enact_ordering_model_name.jsonl'
        
        Args:
            filename: The filename to parse
            
        Returns:
            str: The extracted model name, or None if pattern doesn't match
        """
        basename = Path(filename).name
        if basename.startswith('enact_ordering_') and basename.endswith('.jsonl'):
            model_name = basename[len('enact_ordering_'):-len('.jsonl')]
            return model_name
        return None
    
    def _reset_evaluator_state(self):
        """
        Reset the evaluator state to avoid cross-model contamination
        """
        self.evaluator.eval_results.clear()
        self.evaluator.skipped_items.clear()
        self.evaluator.wrong_case_signatures.clear()
    
    def _save_detailed_results(self, model_name: str):
        """
        Save detailed per-data-point evaluation results.
        
        Args:
            model_name: Name of the model being evaluated
        """
        detailed_output_file = self.detailed_eval_dir / f"enact_ordering_{model_name}.jsonl"
        
        with open(detailed_output_file, 'w') as f:
            for data_id, result in self.evaluator.eval_results.items():
                # Prepare detailed result entry
                detailed_entry = {
                    'id': data_id,
                    'task_name': result.get('task_name'),
                    'type': result.get('type'),
                    'eval_metrics': result.get('eval_metrics', {}),
                    'ground_truth': result.get('gt_answer'),
                    'model_answer': result.get('parsed_answer'),
                    'raw_answer': result.get('raw_answer')
                }
                
                # Add wrong case signatures if available
                if data_id in self.evaluator.wrong_case_signatures:
                    wrong_case_data = self.evaluator.wrong_case_signatures[data_id]
                    detailed_entry['wrong_case_analysis'] = wrong_case_data
                
                json.dump(detailed_entry, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"  Detailed evaluation results saved to: {detailed_output_file}")
    
    def _save_meta_performance(self, model_name: str, overall_report: dict):
        """
        Save meta performance statistics for the model.
        
        Args:
            model_name: Name of the model being evaluated
            overall_report: Overall performance metrics
        """
        meta_output_file = self.meta_performance_dir / f"enact_ordering_{model_name}.json"
        
        # Format the report with world_modeling terminology
        formatted_report = {
            'model_name': model_name,
            'overall_performance': overall_report
        }
        
        with open(meta_output_file, 'w') as f:
            json.dump(formatted_report, f, indent=2)
        
        print(f"  Meta performance results saved to: {meta_output_file}")
    
    def _save_wrong_case_signatures(self, model_name: str):
        """
        Save wrong case signatures to JSONL file.
        
        Args:
            model_name: Name of the model being evaluated
        """
        if not self.analyze_wrong_cases:
            return
        
        signatures_output_file = self.signatures_dir / f"enact_ordering_{model_name}.jsonl"
        
        # Save wrong case results to JSONL file
        with open(signatures_output_file, 'w') as f:
            for data_id, wrong_case_entry in self.evaluator.wrong_case_signatures.items():
                json.dump(wrong_case_entry, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"  Wrong case signatures saved to: {signatures_output_file}")
    
    def _print_performance_summary(self, model_name: str, overall_report: dict):
        """
        Print a formatted performance summary.
        
        Args:
            model_name: Name of the model being evaluated
            overall_report: Overall performance metrics
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY: {model_name}")
        print(f"{'='*60}")
        
        # Overall performance
        print("\nOVERALL PERFORMANCE:")
        if 'overall' in overall_report:
            overall_stats = overall_report['overall']
            print(f"   Total Evaluated: {overall_stats.get('count', 0)}")
            print(f"   Task Accuracy: {overall_stats.get('task_accuracy', 0.0)*100:.2f}%")
            print(f"   Pairwise Accuracy: {overall_stats.get('pairwise_accuracy', 0.0)*100:.2f}%")
        
        # World modeling breakdown
        if 'forward_world_modeling' in overall_report:
            fwd_stats = overall_report['forward_world_modeling']
            print(f"\n   Forward World Modeling:")
            print(f"     Task Accuracy: {fwd_stats.get('task_accuracy', 0.0)*100:.2f}% "
                  f"({fwd_stats.get('count', 0)} samples)")
            print(f"     Pairwise Accuracy: {fwd_stats.get('pairwise_accuracy', 0.0)*100:.2f}%")
        
        if 'inverse_world_modeling' in overall_report:
            inv_stats = overall_report['inverse_world_modeling']
            print(f"\n   Inverse World Modeling:")
            print(f"     Task Accuracy: {inv_stats.get('task_accuracy', 0.0)*100:.2f}% "
                  f"({inv_stats.get('count', 0)} samples)")
            print(f"     Pairwise Accuracy: {inv_stats.get('pairwise_accuracy', 0.0)*100:.2f}%")
    
    def evaluate_single_file(self, jsonl_path: Path, model_name: Optional[str] = None):
        """
        Evaluate a single JSONL file.
        
        Args:
            jsonl_path: Path to the JSONL file
            model_name: Optional model name (if None, extracted from filename)
        """
        if model_name is None:
            model_name = self._extract_model_name_from_filename(jsonl_path)
            if model_name is None:
                print(f"Warning: Could not extract model name from {jsonl_path}, using 'unknown'")
                model_name = "unknown"
        
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*60}")
        print(f"Input file: {jsonl_path}")
        
        try:
            # Clear previous evaluation results
            self._reset_evaluator_state()
            
            # Run the evaluation
            self.evaluator.evaluate(str(jsonl_path), analyze_wrong_case=self.analyze_wrong_cases)
            
            # Generate reports
            overall_report = self.evaluator.report_overall_score()
            
            # Print performance summary
            self._print_performance_summary(model_name, overall_report)
            
            # Save results
            print(f"\nSAVING RESULTS:")
            self._save_meta_performance(model_name, overall_report)
            self._save_detailed_results(model_name)
            if self.analyze_wrong_cases:
                self._save_wrong_case_signatures(model_name)
            
            print(f"\nEvaluation of {model_name} completed successfully!")
            
            return {
                'model_name': model_name,
                'status': 'success',
                'overall_stats': overall_report.get('overall', {})
            }
            
        except Exception as e:
            print(f"ERROR evaluating {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'model_name': model_name,
                'status': 'error',
                'error': str(e)
            }
    
    def get_jsonl_files(self):
        """
        Get all JSONL files from the input directory.
        
        Returns:
            list: List of JSONL file paths
        """
        pattern = str(self.input_path / "enact_ordering_*.jsonl")
        jsonl_files = glob.glob(pattern)
        jsonl_files.sort()
        return [Path(f) for f in jsonl_files]
    
    def process_all_files(self):
        """
        Process file(s) based on the mode.
        For single mode: processes the single file.
        For batch mode: processes all JSONL files in the input directory.
        """
        if self.is_single_mode:
            # Single file mode: process only the input file
            print(f"\nProcessing single file: {self.input_path.name}")
            self.evaluate_single_file(self.input_path)
            print(f"\nProcessing complete!")
            print(f"Results saved to: {self.output_root}")
        else:
            # Batch mode: process all JSONL files
            jsonl_files = self.get_jsonl_files()
            
            if not jsonl_files:
                print(f"No JSONL files found matching pattern 'enact_ordering_*.jsonl' in {self.input_path}")
                return
            
            print(f"\nFound {len(jsonl_files)} files to evaluate:")
            for jsonl_file in jsonl_files:
                model_name = self._extract_model_name_from_filename(jsonl_file)
                print(f"  - {jsonl_file.name} -> {model_name}")
            
            # Process each file
            results_summary = []
            successful_evaluations = 0
            failed_evaluations = 0
            
            for i, jsonl_path in enumerate(jsonl_files, 1):
                print(f"\n[{i}/{len(jsonl_files)}] Processing {jsonl_path.name}...")
                
                result = self.evaluate_single_file(jsonl_path)
                results_summary.append(result)
                
                if result['status'] == 'success':
                    successful_evaluations += 1
                else:
                    failed_evaluations += 1
            
            # Generate final summary
            print("\n" + "="*80)
            print("BATCH EVALUATION SUMMARY")
            print("="*80)
            print(f"Total files processed: {len(results_summary)}")
            print(f"Successful evaluations: {successful_evaluations}")
            print(f"Failed evaluations: {failed_evaluations}")
            
            if successful_evaluations > 0:
                print(f"\nSuccessful evaluations:")
                for result in results_summary:
                    if result['status'] == 'success':
                        task_acc = result['overall_stats'].get('task_accuracy', 0.0)
                        print(f"  ✓ {result['model_name']}: {task_acc*100:.2f}% task accuracy")
            
            if failed_evaluations > 0:
                print(f"\nFailed evaluations:")
                for result in results_summary:
                    if result['status'] == 'error':
                        print(f"  ✗ {result['model_name']}: {result['error']}")
            
            # Save batch summary
            summary_file = self.output_root / "batch_evaluation_summary.json"
            with open(summary_file, 'w') as f:
                json.dump({
                    'total_processed': len(results_summary),
                    'successful': successful_evaluations,
                    'failed': failed_evaluations,
                    'results': results_summary
                }, f, indent=2)
            
            print(f"\nBatch summary saved to: {summary_file}")
            print(f"\nBatch evaluation complete!")
            print(f"Results saved to: {self.output_root}")

