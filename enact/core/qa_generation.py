"""
Question-Answer Generation Pipeline for VLM World Modeling Evaluation.

This module provides the core infrastructure for generating structured question-answer
pairs from robot trajectory data. It supports evaluating Vision-Language Models' world
modeling capabilities through forward and inverse dynamics tasks.
"""

import sys
import json
import random
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np

try:
    from enact.utils.qa_gen_utils import TaskData, QAPair, AbstractQAGenerator
    from enact.core.forward_world_modeling import ForwardWorldModelingGenerator
    from enact.core.inverse_world_modeling import InverseWorldModelingGenerator
    from enact.utils.scene_graph_utils import SceneGraphReader
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class QAGenerationManager:
    """
    Central manager for question-answer generation pipeline.
    
    Orchestrates the entire QA generation process:
    - Loads and parses segmented trajectory data
    - Dispatches tasks to appropriate QA generators
    - Aggregates and saves results to JSONL format
    """

    def __init__(
        self, 
        input_root_dir: str, 
        raw_data_dir: str
    ):
        """
        Initialize the QA generation manager.
        
        Args:
            input_root_dir: Path to directory containing segmented trajectory results
            raw_data_dir: Path to directory containing raw data (scene graphs, images, etc.)
        """
        self.input_root_dir = Path(input_root_dir)
        self.raw_data_dir = Path(raw_data_dir)
        self.task_data_list: List[TaskData] = []
        self.qa_pairs: List[QAPair] = []
        
        if not self.input_root_dir.exists():
            raise ValueError(f"Output root directory does not exist: {input_root_dir}")
        
        self._load_all_tasks()

    def _load_all_tasks(self):
        """
        Load all task data from segmented and raw data directories.
        
        Parses task directories, extracts key frame IDs from segmented scene graphs,
        and prepares TaskData objects for QA generation.
        """
        print(f"Loading segmented tasks from: {self.input_root_dir}")
        print(f"Loading raw data from: {self.raw_data_dir}")
        
        task_dirs = sorted([d for d in self.input_root_dir.iterdir() if d.is_dir()])
        
        for task_dir in task_dirs:
            task_name = task_dir.name
            segmented_scene_graph_file = task_dir / "segmented_scene_graph_0.json"
            raw_data_task_dir = self.raw_data_dir / task_name
            raw_scene_graph_file = raw_data_task_dir / "scene_graph_0.json"
            
            if not segmented_scene_graph_file.exists():
                print(f"Warning: No segmented scene graph file found for task {task_name}, skipping...")
                continue

            if not raw_scene_graph_file.exists():
                print(f"Warning: No raw scene graph file found for task {task_name}, skipping...")
                continue
            
            try:
                segmented_scene_graph_reader = SceneGraphReader(str(segmented_scene_graph_file))
                key_frame_ids = segmented_scene_graph_reader.get_available_frame_ids()

                raw_scene_graph_reader = SceneGraphReader(str(raw_scene_graph_file))
                image_root_path, image_paths = self._collect_image_paths(raw_data_task_dir)
                
                task_data = TaskData(
                    task_name=task_name,
                    scene_graph_reader=raw_scene_graph_reader,
                    key_frame_ids=key_frame_ids,
                    image_paths=image_paths,
                    task_dir=str(task_dir),
                    image_root_path=image_root_path
                )
                
                self.task_data_list.append(task_data)
                print(f"Loaded task: {task_name} with {len(key_frame_ids)} key frames")
                
            except Exception as e:
                print(f"Error loading task {task_name}: {str(e)}")
                continue

    def _collect_image_paths(
        self, 
        task_dir: Path, 
        key_frame_ids=None
    ) -> Tuple[str, Dict[str, Dict[str, str]]]:
        """
        Collect image paths for all frames and sensors.
        
        Args:
            task_dir: Path to the task directory
            key_frame_ids: Optional list of specific frame IDs to collect. If None, collects all available frames.
            
        Returns:
            Tuple containing:
                - image_root_path: Root path for images
                - image_paths: Nested dict mapping frame_id -> sensor_name -> image_path
        """
        image_paths = {}
        sensor_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith('external_sensor')]
        image_root_path = task_dir.parent

        if key_frame_ids is None:
            all_frame_ids = set()
            for sensor_dir in sensor_dirs:
                for image_file in sensor_dir.glob("*.png"):
                    frame_id = image_file.stem
                    try:
                        frame_id_int = int(frame_id)
                        all_frame_ids.add(str(frame_id_int))
                    except ValueError:
                        continue
            
            key_frame_ids = sorted(all_frame_ids, key=int)
        
        for frame_id in key_frame_ids:
            image_paths[frame_id] = {}
            
            for sensor_dir in sensor_dirs:
                sensor_name = sensor_dir.name
                image_file = sensor_dir / f"{int(frame_id):05d}.png"
                
                if image_file.exists():
                    image_paths[frame_id][sensor_name] = str(image_file)
        
        return image_root_path, image_paths

    def generate(
        self, 
        qa_type: str, 
        step_length=None, 
        flush_to_file=None, 
        seed: int=42, 
        num_to_sample: int=30
    ) -> List[QAPair]:
        """
        Generate QA pairs of the specified type for all loaded tasks.
        
        Args:
            qa_type: Type of QA to generate ("forward" or "inverse")
            step_length: Optional step length for multi-step generators
            flush_to_file: If provided, stream QA pairs to this file path instead of storing in memory
            seed: Random seed for reproducible results
            num_to_sample: Number of sequences to sample for multi-step generators
            
        Returns:
            List of generated QA pairs, or task statistics dict if flushing to file
        """
        random.seed(seed)
        np.random.seed(seed)
        print(f"Set random seeds to {seed} for deterministic QA generation")
        
        generator_class = self._get_generator_class(qa_type)
        if generator_class is None:
            raise ValueError(f"No generator found for Q&A type: {qa_type}")
        
        if step_length is not None:
            generator = generator_class(step_length=step_length)
        else:
            generator = generator_class()
        
        generated_pairs = []
        task_stats = {}
        
        print(f"Generating {qa_type} Q&A pairs for {len(self.task_data_list)} tasks...")
        
        for task_data in self.task_data_list:
            try:
                if qa_type in ["forward", "inverse"]:
                    task_pairs = generator.generate(task_data, num_to_sample=num_to_sample)
                else:
                    task_pairs = generator.generate(task_data)
                
                if flush_to_file:
                    self._append_to_jsonl(flush_to_file, task_pairs)
                    task_stats[task_data.task_name] = len(task_pairs)
                    print(f"Flushed {len(task_pairs)} Q&A pairs for task: {task_data.task_name}")
                else:
                    generated_pairs.extend(task_pairs)
                    print(f"Generated {len(task_pairs)} Q&A pairs for task: {task_data.task_name}")
                    
            except Exception as e:
                import traceback
                print(f"Full traceback:")
                traceback.print_exc()
                print(f"Error generating Q&A for task {task_data.task_name}: {str(e)}")
                if flush_to_file:
                    task_stats[task_data.task_name] = 0
                continue
        
        if not flush_to_file:
            self.qa_pairs.extend(generated_pairs)
        
        total_pairs = len(generated_pairs) if not flush_to_file else sum(task_stats.values())
        print(f"Total generated Q&A pairs: {total_pairs}")
        
        if flush_to_file:
            return task_stats
        return generated_pairs

    def _get_generator_class(self, qa_type: str) -> AbstractQAGenerator:
        """
        Get the generator class for the specified QA type.
        
        Args:
            qa_type: Type of QA generator ("forward" or "inverse")
            
        Returns:
            Generator class for the specified type, or None if not found
        """
        if qa_type == "forward":
            return ForwardWorldModelingGenerator
        elif qa_type == "inverse":
            return InverseWorldModelingGenerator
        else:
            return None

    def save_to_jsonl(self, output_path: str, append_mode: bool = False):
        """
        Save all generated QA pairs to a JSONL file.
        
        Args:
            output_path: Path to the output JSONL file
            append_mode: If True, append to existing file instead of overwriting
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        mode = 'a' if append_mode else 'w'
        with open(output_path, mode, encoding='utf-8') as f:
            for qa_pair in self.qa_pairs:
                json.dump(qa_pair.to_dict(), f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Saved {len(self.qa_pairs)} Q&A pairs to: {output_path}")

    def _append_to_jsonl(self, output_path: str, qa_pairs: List[QAPair]):
        """
        Append QA pairs to a JSONL file immediately.
        
        Args:
            output_path: Path to the output JSONL file
            qa_pairs: QA pairs to append
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'a', encoding='utf-8') as f:
            for qa_pair in qa_pairs:
                json.dump(qa_pair.to_dict(), f, ensure_ascii=False)
                f.write('\n')

    def clear_qa_pairs(self):
        """Clear all stored Q&A pairs."""
        self.qa_pairs.clear()

    @property
    def num_tasks(self) -> int:
        """Return the number of loaded tasks."""
        return len(self.task_data_list)

    @property
    def num_qa_pairs(self) -> int:
        """Return the number of stored Q&A pairs."""
        return len(self.qa_pairs)
