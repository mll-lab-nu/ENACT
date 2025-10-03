"""
Inverse World Modeling Q&A Generator.

This module generates multi-step inverse world modeling questions where users must
determine the correct sequence of actions that occurred given a series of observed states.
"""

import sys
import random
from typing import Dict, List, Any
from pathlib import Path
from tqdm import tqdm
import numpy as np
# Add PIL imports for image processing
from PIL import Image, ImageDraw, ImageFont
import hashlib

try:
    from enact.utils.qa_gen_utils import TaskData, QAPair, AbstractQAGenerator
    from enact.utils.state_change_translator import StateChangeTranslator
    from enact.utils.qa_prompt_template import multi_inv_ordering_prompt
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class InverseWorldModelingGenerator(AbstractQAGenerator):
    """
    Generator for multi-step inverse world modeling QA pairs.
    
    Creates questions that present a sequence of observed states and ask users to
    determine the correct order of actions that produced those state transitions.
    """

    def __init__(
        self, 
        visual_prompt: bool = True, 
        step_length: int = 5
    ):
        """
        Initialize the inverse world modeling generator.
        
        Args:
            visual_prompt: Whether to add visual labels to images
            step_length: Number of frames in each sequence (must be between 3 and 10)
        """
        
        self.translator = StateChangeTranslator(type="multi_inverse_world_modeling")
        self.visual_prompt = visual_prompt
        self.step_length = step_length
        assert 3 <= self.step_length <= 10, "Step length for inverse world modeling should be between 3 and 10."
        self.sensor_names = ["external_sensor1"]

    @property
    def qa_type(self) -> str:
        return f"inverse_world_modeling_{self.step_length}_steps"

    def visual_prompt_path(self, image_root_dir) -> str:
        """
        Get the path for storing visual prompt images.
        
        Args:
            image_root_dir: Root directory for images
            
        Returns:
            Path to the visual prompt directory
        """
        return image_root_dir / 'QA' / 'images' / self.qa_type
        
    def _has_meaningful_changes(self, diff: Dict[str, Any]) -> bool:
        """
        Check if a diff contains meaningful changes worth asking about.
        
        Args:
            diff: Scene graph difference
            
        Returns:
            bool: True if changes are meaningful
        """
        if diff.get('type') == 'empty':
            return False
        
        # Check for any substantial changes
        for operation in ['add', 'remove', 'update']:
            if operation in diff:
                # Node changes are always meaningful
                if diff[operation].get('nodes'):
                    return True
                
                # Edge changes are meaningful if not just Contact states
                for edge in diff[operation].get('edges', []):
                    states = edge.get('states', [])
                    non_contact_states = [s for s in states if 'Contact' not in s]
                    if non_contact_states:
                        return True
        
        return False
    
    def _is_valid_transition(
        self, 
        frame_a_id: str, 
        frame_b_id: str, 
        task_data: TaskData
    ) -> bool:
        """
        Check if a transition between two frames is valid.
        
        A transition is valid if it contains meaningful changes (1-5 state change sentences).
        
        Args:
            frame_a_id: Starting frame ID
            frame_b_id: Ending frame ID
            task_data: Task data containing scene graphs
            
        Returns:
            True if the transition is valid
        """
        visible_diff = task_data.scene_graph_reader.get_visible_full_diff(
            frame_a_id, frame_b_id, self.sensor_names, partial_diff=True
        )
        if visible_diff.get('type') == 'empty' or not self._has_meaningful_changes(visible_diff):
            return False
        
        gt_desc = self.translator.translate_diff(visible_diff)
        total_diff = gt_desc.count(".") if gt_desc else 0
        
        return 1 <= total_diff <= 5
    
    def _build_valid_transitions_graph(
        self, 
        key_frame_ids: List[str], 
        task_data: TaskData
    ) -> Dict[str, List[str]]:
        """
        Build a directed graph of all valid frame transitions (Phase 1).
        
        Computes all valid transitions between frames and stores them as an adjacency list.
        This is an O(N²) operation but is performed only once per task.
        
        Args:
            key_frame_ids: List of frame IDs to consider
            task_data: Task data containing scene graphs
            
        Returns:
            Dictionary mapping each frame ID to a list of valid successor frame IDs
        """
        num_frames = len(key_frame_ids)
        graph = {frame_id: [] for frame_id in key_frame_ids}
        
        print("Phase 1: Building valid transitions graph...")

        for i in tqdm(range(num_frames), desc="Building Graph"):
            for j in range(i + 1, num_frames):
                frame_a = key_frame_ids[i]
                frame_b = key_frame_ids[j]
                if self._is_valid_transition(frame_a, frame_b, task_data):
                    graph[frame_a].append(frame_b)
        return graph

    def _count_paths_with_dp(
        self, 
        graph: Dict[str, List[str]], 
        key_frame_ids: List[str]
    ) -> np.ndarray:
        """
        Count all valid paths using Dynamic Programming (Phase 2).
        
        Uses DP to count the number of valid paths of each length ending at each frame.
        dp_table[k][i] stores the number of valid paths of length (k+1) ending at frame i.
        
        Args:
            graph: Adjacency list of valid transitions
            key_frame_ids: List of frame IDs
            
        Returns:
            Numpy array where dp_table[k][i] is the count of paths of length (k+1) ending at frame i
        """
        num_frames = len(key_frame_ids)
        dp_table = np.zeros((self.step_length, num_frames), dtype=np.int64)

        # Base case: each frame is a valid path of length 1
        dp_table[0, :] = 1

        print("Phase 2: Counting valid paths with Dynamic Programming...")
        
        for k in tqdm(range(1, self.step_length), desc="DP Path Counting"):
            for i in range(num_frames):
                current_frame = key_frame_ids[i]
                # Count paths by summing over all valid predecessors
                for j in range(i):
                    predecessor_frame = key_frame_ids[j]
                    if current_frame in graph[predecessor_frame]:
                        dp_table[k, i] += dp_table[k - 1, j]
        
        return dp_table

    def _sample_paths_randomly(
        self,
        num_to_sample: int,
        graph: Dict[str, List[str]],
        dp_table: np.ndarray,
        key_frame_ids: List[str]
    ) -> List[List[str]]:
        """
        Sample unique paths using weighted random backtracking (Phase 3).
        
        Samples paths by randomly selecting end frames (weighted by path count) and
        backtracking through predecessors. Each step in the backtracking is weighted
        by the number of paths through that predecessor.
        
        Args:
            num_to_sample: Number of unique paths to sample.
            graph: Adjacency list of valid transitions.
            dp_table: DP table from _count_paths_with_dp.
            key_frame_ids: List of frame IDs.
            
        Returns:
            List of unique sampled paths, where each path is a list of frame IDs.
        """
        sampled_sequences = []
        seen_paths = set()
        num_frames = len(key_frame_ids)
        final_k = self.step_length - 1

        # Weight each end frame by the number of paths ending at it
        end_node_population = list(range(num_frames))
        end_node_weights = dp_table[final_k, :]
        
        total_weight = np.sum(end_node_weights)
        if total_weight == 0:
            return []
        
        print(f"Phase 3: Sampling {num_to_sample} unique paths using Weighted Random Backtracking...")

        def _get_one_random_path(start_node_idx: int) -> List[str]:
            """Reconstruct one random path by backtracking from the end frame."""
            path_reversed = [key_frame_ids[start_node_idx]]
            current_idx = start_node_idx
            
            # Backtrack from the end to the beginning
            for k in range(final_k, 0, -1):
                # Find all valid predecessors and their weights
                predecessors = []
                weights = []
                for prev_idx in range(current_idx):
                    prev_frame = key_frame_ids[prev_idx]
                    current_frame = key_frame_ids[current_idx]
                    if current_frame in graph[prev_frame] and dp_table[k - 1, prev_idx] > 0:
                        predecessors.append(prev_idx)
                        weights.append(dp_table[k - 1, prev_idx])
                
                if not predecessors:
                    break
                
                # Choose predecessor weighted by the number of paths through it
                chosen_predecessor_idx = random.choices(predecessors, weights=weights, k=1)[0]
                path_reversed.append(key_frame_ids[chosen_predecessor_idx])
                current_idx = chosen_predecessor_idx
                
            return list(reversed(path_reversed))

        # Main sampling loop with duplicate avoidance
        max_attempts = num_to_sample * 10
        attempts = 0
        
        with tqdm(total=num_to_sample, desc="Sampling Unique Paths") as pbar:
            while len(sampled_sequences) < num_to_sample and attempts < max_attempts:
                end_node_idx = random.choices(end_node_population, weights=end_node_weights, k=1)[0]
                
                path = _get_one_random_path(end_node_idx)
                if len(path) == self.step_length:
                    path_tuple = tuple(path)
                    if path_tuple not in seen_paths:
                        seen_paths.add(path_tuple)
                        sampled_sequences.append(path)
                        pbar.update(1)
                
                attempts += 1
        
        if len(sampled_sequences) < num_to_sample:
            print(f"Warning: Only found {len(sampled_sequences)} unique paths after {attempts} attempts.")

        return sampled_sequences
    
    def _translate_sequence_to_actions(self, task_data: TaskData, sequence: List[str]) -> List[str]:
        """
        Translate a sequence of frames into action descriptions.
        
        Args:
            task_data: Task data containing scene graphs
            sequence: List of frame IDs in sequential order
            
        Returns:
            List of action descriptions, one for each frame transition
        """
        action_descriptions = []
        for i in range(len(sequence) - 1):
            frame_a_id = sequence[i]
            frame_b_id = sequence[i+1]
            diff = task_data.scene_graph_reader.get_visible_full_diff(
                frame_a_id, frame_b_id, self.sensor_names, partial_diff=True
            )
            action_desc = self.translator.translate_diff(diff)
            if not action_desc:
                action_desc = "No meaningful change is observed."
            action_descriptions.append(action_desc)
        return action_descriptions
    

    def generate(self, task_data: TaskData, num_to_sample: int=30, max_qa_num: int=25) -> List[QAPair]:
        """
        Generate inverse world modeling QA pairs for a task.
        
        This method:
        1. Builds a graph of valid frame transitions (Phase 1)
        2. Counts all valid paths using dynamic programming (Phase 2)
        3. Samples unique paths using weighted random backtracking (Phase 3)
        4. Creates QA pairs from the sampled sequences (Phase 4)
        
        Args:
            task_data: Task data containing scene graphs and images
            num_to_sample: Number of unique sequences to sample
            max_qa_num: Maximum number of QA pairs to return
            
        Returns:
            List of generated QA pairs
        """
        # Set seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        key_frame_ids = sorted(task_data.key_frame_ids, key=int)
        if len(key_frame_ids) < self.step_length:
            return []

        # Phase 1: Build the transition graph
        graph = self._build_valid_transitions_graph(key_frame_ids, task_data)
        
        # Phase 2: Count paths using dynamic programming
        dp_table = self._count_paths_with_dp(graph, key_frame_ids)
        total_paths = dp_table[self.step_length - 1].sum()
        
        print(f"\nDP table computed. Found a total of {total_paths} valid sequences.")
        
        if total_paths == 0:
            return []
        
        # Phase 3: Sample unique paths with weighted random backtracking
        actual_num_to_sample = min(num_to_sample, int(total_paths))
        all_valid_sequences = self._sample_paths_randomly(
            actual_num_to_sample, graph, dp_table, key_frame_ids
        )
        
        print(f"\nSuccessfully sampled {len(all_valid_sequences)} representative sequences.")
        
        # Phase 4: Generate QA pairs from sampled sequences
        print(f"Phase 4: Generating QA pairs from {len(all_valid_sequences)} sequences...")
        
        qa_pairs = []
        for seq in tqdm(all_valid_sequences, desc="Generating Ordering QA pairs"):
            try:
                qa_pair = self._create_ordering_qa_pair(task_data, seq)
                if qa_pair:
                    qa_pairs.append(qa_pair)
            except Exception as e:
                import traceback
                print(f"Error generating QA for sequence {seq}: {e}")
                traceback.print_exc()
                continue
        
        print(f"\nGenerated {len(qa_pairs)} multi-step inverse world modeling QA pairs.")
        
        if max_qa_num:
            print(f"Truncating to {max_qa_num} QA pairs.")
            qa_pairs = random.sample(qa_pairs, min(max_qa_num, len(qa_pairs)))
        
        return qa_pairs
    
    def _add_text_to_image(self, image_path: str, text: str, output_path: str) -> None:
        """
        Add a text label to an image and save it.
        
        Args:
            image_path: Path to the source image
            text: Text to add to the image
            output_path: Path where the labeled image will be saved
        """
        try:
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            
            # Configure font size based on image height
            font_size = max(40, img.height // 10)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
            except (OSError, IOError):
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except (OSError, IOError):
                    font = ImageFont.load_default()
            
            # Text styling
            text_color = (255, 20, 20)
            outline_color = (255, 255, 255)
            outline_width = 3
            
            # Position at top-left with padding
            x, y = 15, 15
            
            # Draw outline
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
            
            # Draw main text
            draw.text((x, y), text, font=font, fill=text_color)
            
            img.save(output_path)
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            import shutil
            shutil.copy2(image_path, output_path)

    def generate_qa_id_hash(self, task_name: str, qa_type: str, sequence: List[str]) -> str:
        """
        Generate a unique hash-based ID for a QA pair.
        
        Args:
            task_name: Name of the task
            qa_type: Type of QA (e.g., 'inverse_world_modeling_5_steps')
            sequence: Sequence of frame IDs
            
        Returns:
            Unique ID string in format: {task_name}_{qa_type}_{hash}
        """
        sequence_str = '_'.join(sequence)
        sequence_hash = hashlib.sha256(sequence_str.encode()).hexdigest()[:8]
        return f"{task_name}_{qa_type}_{sequence_hash}"
    
    def _create_ordering_qa_pair(
        self,
        task_data: TaskData,
        correct_sequence: List[str]
    ) -> QAPair:
        """
        Create a QA pair for the inverse world modeling ordering task.
        
        Given a sequence of observed states, this creates a question asking the user
        to order a set of shuffled actions that occurred between those states.
        
        Args:
            task_data: Task data containing scene graphs and images
            correct_sequence: List of frame IDs in the correct temporal order
            
        Returns:
            A QAPair object
        """
        assert len(correct_sequence) > 2, "Correct sequence must be at least 3 frames long"

        qa_id = self.generate_qa_id_hash(task_data.task_name, self.qa_type, correct_sequence)
        sensor_name = self.sensor_names[0]
        image_paths = [task_data.image_paths[frame_id][sensor_name] for frame_id in correct_sequence]
        start_image_path = image_paths[0]
        next_states = correct_sequence[1:]

        # Create visual prompts with labels
        final_start_image = start_image_path
        final_next_state_images = []
        
        if self.visual_prompt:
            image_root_dir = task_data.image_root_path.parent
            new_base_dir = self.visual_prompt_path(image_root_dir)
            output_dir = Path(new_base_dir) / task_data.task_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Add label to current state image
            cur_state_output_path = output_dir / f"{qa_id}_cur_state.png"
            self._add_text_to_image(start_image_path, "Current State", str(cur_state_output_path))
            final_start_image = str(cur_state_output_path)

            # Add labels to next state images
            for i, frame_id in enumerate(next_states):
                next_state_image_path = task_data.image_paths[frame_id][sensor_name]
                label = f"Next State {i+1}"
                next_state_output_path = output_dir / f"{qa_id}_next_state_{i+1}.png"
                self._add_text_to_image(next_state_image_path, label, str(next_state_output_path))
                final_next_state_images.append(str(next_state_output_path))
        else:
            # Use original images without labels
            for frame_id in next_states:
                image_path = task_data.image_paths[frame_id][sensor_name]
                final_next_state_images.append(image_path)

        # Prepare all images with relative paths
        all_images = [final_start_image] + final_next_state_images
        all_images = [str(Path(image_path).relative_to(image_root_dir)) for image_path in all_images]
        
        # Generate action descriptions and shuffle them
        correct_action_sequence = self._translate_sequence_to_actions(task_data, correct_sequence)
        shuffled_action_sequence = correct_action_sequence[:]
        random.shuffle(shuffled_action_sequence)

        # Find the correct order (1-indexed positions)
        correct_order = []
        used_positions = set()
        for original_action in correct_action_sequence:
            for i, shuffled_action in enumerate(shuffled_action_sequence):
                if shuffled_action == original_action and i not in used_positions:
                    shuffled_position = i + 1
                    correct_order.append(shuffled_position)
                    used_positions.add(i)
                    break
            else:
                raise ValueError(f"Could not find unused position for action: {original_action}")

        # Format shuffled actions with numbered labels
        numbered_shuffled_actions = [f"[Action {i+1}] {action}" for i, action in enumerate(shuffled_action_sequence)]
        numbered_action = '\n'.join(numbered_shuffled_actions)
        question = multi_inv_ordering_prompt.format(SHUFFLED_ACTIONS=numbered_action)

        # Validate correct_order sequence
        expected_sequence = list(range(1, len(correct_action_sequence) + 1))
        if sorted(correct_order) != expected_sequence:
            raise ValueError(f"Invalid correct_order sequence: {correct_order}. "
                           f"Expected sorted: {expected_sequence}, got sorted: {sorted(correct_order)}")

        # Create the ground truth answer
        gt_answer = {
            "type": self.qa_type,
            "options": [],
            "correct_option": correct_order,
        }

        qa_pair = QAPair(
            id=qa_id,
            images=all_images,
            task_name=task_data.task_name,
            key_frame_ids=correct_sequence,
            question=question,
            gt_answer=gt_answer
        )
        
        return qa_pair
