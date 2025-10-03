#!/usr/bin/env python3
"""
Segmentation Processor

Processes task directories for scene graph segmentation and frame extraction.
Supports both single task mode (input is a specific task directory) and 
batch mode (input is a root directory containing multiple task directories).

Uses the FrameSegmentManager class to perform scene graph segmentation and 
extract relevant frames into a corresponding output directory structure.
"""

import sys
import shutil
from pathlib import Path

try:
    from enact.core.segmentation import FrameSegmentManager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class SegmentationProcessor:
    """
    Processes task directories for frame segmentation and extraction.
    Supports both single task and batch processing modes.
    """
    
    def __init__(self, input_root: str, output_root: str):
        """
        Initialize the segmentation processor.
        
        Args:
            input_root: For batch mode: root directory containing multiple task directories.
                       For single mode: path to a specific task directory.
            output_root: Root directory for output structure
        """
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        
        if not self.input_root.exists():
            raise FileNotFoundError(f"Input root directory not found: {input_root}")
        
        # Create output root if it doesn't exist
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Determine processing mode based on input structure
        self.is_single_mode = self._is_single_task_directory(self.input_root)
        
        if self.is_single_mode:
            print(f"Single task mode detected: {self.input_root.name}")
        else:
            print(f"Batch mode detected: processing multiple tasks")
    
    def _is_single_task_directory(self, path: Path) -> bool:
        """
        Check if the given path is a single task directory.
        
        Args:
            path: Path to check
            
        Returns:
            bool: True if it's a single task directory, False if it's a batch root
        """
        # A single task directory should contain scene_graph_0.json
        scene_graph_file = path / "scene_graph_0.json"
        return scene_graph_file.exists()
    
    def get_task_directories(self):
        """
        Get all task directories from the input root.
        
        Returns:
            list: List of task directory paths
        """
        task_dirs = []
        for item in self.input_root.iterdir():
            if item.is_dir():
                # Check if directory contains scene_graph_0.json (indicating it's a task directory)
                scene_graph_file = item / "scene_graph_0.json"
                if scene_graph_file.exists():
                    task_dirs.append(item)
        
        task_dirs.sort()  # Sort for consistent processing order
        return task_dirs
    
    def validate_task_directory(self, task_dir: Path) -> bool:
        """
        Validate that a task directory has the required structure.
        
        Args:
            task_dir: Path to the task directory
            
        Returns:
            bool: True if valid, False otherwise
        """
        scene_graph_file = task_dir / "scene_graph_0.json"
        external_sensor1 = task_dir / "external_sensor1"
        
        if not scene_graph_file.exists():
            print(f"Warning: {task_dir.name} missing scene_graph_0.json")
            return False
        
        if not external_sensor1.exists():
            print(f"Warning: {task_dir.name} missing sensor directories")
            return False
        
        return True
    
    def copy_extracted_frames(self, source_sensor_dir: Path, target_sensor_dir: Path, 
                            extracted_frames: list):
        """
        Copy extracted frames from source to target sensor directory.
        
        Args:
            source_sensor_dir: Source sensor directory path
            target_sensor_dir: Target sensor directory path
            extracted_frames: List of frame IDs to copy
        """
        target_sensor_dir.mkdir(parents=True, exist_ok=True)
        
        copied_count = 0
        for frame_id in extracted_frames:
            # Convert frame ID to filename format (5-digit with .png extension)
            frame_filename = f"{int(frame_id):05d}.png"
            source_file = source_sensor_dir / frame_filename
            target_file = target_sensor_dir / frame_filename
            
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                copied_count += 1
            else:
                print(f"Warning: Frame {frame_filename} not found in {source_sensor_dir}")
        
        print(f"  Copied {copied_count}/{len(extracted_frames)} frames to {target_sensor_dir.name}")
    
    def process_task_directory(self, task_dir: Path):
        """
        Process a single task directory.
        
        Args:
            task_dir: Path to the task directory to process
        """
        print(f"\nProcessing {task_dir.name}...")
        
        # Validate task directory structure
        if not self.validate_task_directory(task_dir):
            print(f"Skipping {task_dir.name} due to missing files")
            return
        
        # Create output task directory
        output_task_dir = self.output_root / task_dir.name
        output_task_dir.mkdir(parents=True, exist_ok=True)
        
        # Process scene graph segmentation
        scene_graph_path = task_dir / "scene_graph_0.json"
        
        try:
            # Initialize FrameSegmentManager
            manager = FrameSegmentManager(str(scene_graph_path))
            
            # Extract changes
            changes = manager.extract_changes(method="cosine_similarity")
            
            # Save segmented scene graph
            output_scene_graph = output_task_dir / "segmented_scene_graph_0.json"
            manager.save_changes(changes, str(output_scene_graph))
            
            # Get extracted frame IDs
            extracted_frames = manager.extracted_frames
            print(f"  Extracted {len(extracted_frames)} frames")
            
            source_sensor1 = task_dir / "external_sensor1"
            target_sensor1 = output_task_dir / "external_sensor1"
            
            self.copy_extracted_frames(source_sensor1, target_sensor1, extracted_frames)
            
            print(f"  Successfully processed {task_dir.name}")
            
        except Exception as e:
            print(f"Error processing {task_dir.name}: {e}")
            import traceback
            traceback.print_exc()
    
    def process_all_tasks(self):
        """
        Process task directory/directories based on the mode.
        For single mode: processes the single task directory.
        For batch mode: processes all task directories in the input root.
        """
        if self.is_single_mode:
            # Single task mode: process only the input directory itself
            print(f"\nProcessing single task: {self.input_root.name}")
            self.process_task_directory(self.input_root)
            print(f"\nProcessing complete!")
            print(f"Results saved to: {self.output_root}")
        else:
            # Batch mode: process all task directories
            task_dirs = self.get_task_directories()
            
            if not task_dirs:
                print("No task directories found in input root")
                return
            
            print(f"Found {len(task_dirs)} task directories to process")
            
            for task_dir in task_dirs:
                self.process_task_directory(task_dir)
            
            print(f"\nBatch processing complete!")
            print(f"Results saved to: {self.output_root}")