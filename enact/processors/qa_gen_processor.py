#!/usr/bin/env python3
"""
Batch QA Generation Processor

Processes multiple task directories from segmented trajectories and generates
question-answer pairs using the QAGenerationManager class, saving results to JSONL format.

Key parameters:
- seed: Random seed for reproducible results (default: 42)
- num_to_sample: Number of sequences to sample for multi-step generators (default: 30)
"""

import sys
import json
from pathlib import Path

try:
    from enact.core.qa_generation import QAGenerationManager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class BatchQAProcessor:
    """
    Batch processor for generating QA pairs from multiple task directories.
    
    Uses QAGenerationManager to process segmented trajectories and generate
    forward and inverse world modeling QA pairs across multiple step lengths.
    """
    
    def __init__(
        self, 
        input_root: str, 
        output_file: str, 
        raw_data_dir: str, 
        seed: int = 42, 
        num_to_sample: int = 30
    ):
        """
        Initialize the batch QA processor.
        
        Args:
            input_root: Root directory containing segmented task directories
            output_file: Output JSONL file path
            raw_data_dir: Root directory containing raw data (scene graphs, images, etc.)
            seed: Random seed for reproducible results
            num_to_sample: Number of sequences to sample for multi-step generators
        """
        self.input_root = Path(input_root)
        self.output_file = Path(output_file)
        self.raw_data_dir = Path(raw_data_dir)
        self.seed = seed
        self.num_to_sample = num_to_sample
        
        if not self.input_root.exists():
            raise FileNotFoundError(f"Input root directory not found: {input_root}")
        
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def get_task_directories(self):
        """
        Get all valid task directories from the input root.
        
        Returns:
            List of task directory paths sorted alphabetically
        """
        task_dirs = []
        for item in self.input_root.iterdir():
            if item.is_dir():
                scene_graph_file = item / "segmented_scene_graph_0.json"
                if scene_graph_file.exists():
                    task_dirs.append(item)
        
        task_dirs.sort()
        return task_dirs
    
    def run(self):
        """
        Run the complete batch QA generation process using QAGenerationManager.
        """
        print("=== BATCH QA GENERATION ===")
        print(f"Input directory: {self.input_root}")
        print(f"Output file: {self.output_file}")
        print()
        
        try:
            manager = QAGenerationManager(str(self.input_root), str(self.raw_data_dir))
            
            print(f"Loaded {manager.num_tasks} tasks from {self.input_root}")
            
            if manager.num_tasks == 0:
                print("No valid tasks found for QA generation")
                return
            
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            if self.output_file.exists():
                self.output_file.unlink()
            
            step_numbers = [3, 4, 5, 6, 7, 8, 9, 10]
            qa_report = {}
            task_names = [task_data.task_name for task_data in manager.task_data_list]
            for task_name in task_names:
                qa_report[task_name] = {}
                for step_num in step_numbers:
                    qa_report[task_name][step_num] = {
                        'forward': 0,
                        'inverse': 0,
                        'total': 0
                    }
            
            for step_num in step_numbers:
                print(f"\n{'='*50}")
                print(f"Processing Step {step_num}")
                print(f"{'='*50}")
                
                print(f"\nGenerating Forward QA pairs (step={step_num}, samples={self.num_to_sample})...")
                forward_stats = manager.generate("forward", step_length=step_num, flush_to_file=str(self.output_file), seed=self.seed, num_to_sample=self.num_to_sample)
                
                print(f"\nGenerating Inverse QA pairs (step={step_num}, samples={self.num_to_sample})...")
                inverse_stats = manager.generate("inverse", step_length=step_num, flush_to_file=str(self.output_file), seed=self.seed, num_to_sample=self.num_to_sample)
            
                for task_name in task_names:
                    forward_count = forward_stats.get(task_name, 0)
                    inverse_count = inverse_stats.get(task_name, 0)
                    
                    qa_report[task_name][step_num]['forward'] = forward_count
                    qa_report[task_name][step_num]['inverse'] = inverse_count
                    qa_report[task_name][step_num]['total'] = forward_count + inverse_count
                
                step_forward_total = sum(forward_stats.values())
                step_inverse_total = sum(inverse_stats.values())
                step_total = step_forward_total + step_inverse_total
                
                print(f"Step {step_num} Summary: Forward={step_forward_total}, Inverse={step_inverse_total}, Total={step_total}")
            
            print(f"\nBATCH QA GENERATION COMPLETE!")
            print("=" * 60)
            print(f"FINAL QA GENERATION REPORT:")
            print("=" * 60)
            
            overall_forward = 0
            overall_inverse = 0
            overall_total = 0
            
            for step_num in step_numbers:
                step_forward = sum(qa_report[task_name][step_num]['forward'] for task_name in task_names)
                step_inverse = sum(qa_report[task_name][step_num]['inverse'] for task_name in task_names)
                step_total = step_forward + step_inverse
                
                overall_forward += step_forward
                overall_inverse += step_inverse
                overall_total += step_total
                
                print(f"Step {step_num:2d}: Forward={step_forward:4d}, Inverse={step_inverse:4d}, Total={step_total:4d}")
            
            print("-" * 60)
            print(f"OVERALL: Forward={overall_forward:4d}, Inverse={overall_inverse:4d}, Total={overall_total:4d}")
            print(f"All QA pairs saved to: {self.output_file}")
            print("=" * 60)
            
            print(f"\nDetailed QA Report by Task and Step:")
            print("Format: report[task_name][step_length][qa_type] = count")
            print("-" * 60)
            print(json.dumps(qa_report, indent=2))
            
        except Exception as e:
            print(f"Error during QA generation: {e}")
            import traceback
            traceback.print_exc()
            raise
