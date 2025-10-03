"""
Forward World Modeling Q&A Generator.

This module implements the ForwardWorldModelingGenerator class that generates 
multi-step forward world modeling questions. Given an initial state and a sequence of 
actions, the task is to determine the correct ordering of future states.
"""
import sys
import random
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
from tqdm import tqdm
import hashlib
from PIL import Image, ImageDraw, ImageFont

try:
    from enact.utils.qa_gen_utils import TaskData, QAPair, AbstractQAGenerator
    from enact.utils.qa_prompt_template import multi_fwd_ordering_prompt
    from enact.utils.state_change_translator import StateChangeTranslator
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class ForwardWorldModelingGenerator(AbstractQAGenerator):
    """
    Generator for multi-step forward world modeling QA pairs.

    Given an initial state and a sequence of actions, this generator creates
    questions that ask users to order a set of shuffled future states correctly.
    """

    def __init__(
        self, 
        visual_prompt: bool=True, 
        step_length: int=5
    ):
        """
        Initialize the forward world modeling generator.

        Args:
            visual_prompt: Whether to add visual labels to images.
            step_length: Number of frames in each generated sequence (must be between 3 and 10).
        """
        # Set seeds for reproducibility at initialization
        random.seed(42)
        np.random.seed(42)
        
        self.translator = StateChangeTranslator(
            type='multi_forward_world_modeling'
        )
        self.visual_prompt = visual_prompt
        self.step_length = step_length
        assert 3 <= self.step_length <= 10, f"Step length should be between 3 and 10. Got {self.step_length} instead."
        self.sensor_names = ['external_sensor1']
    
    @property
    def qa_type(self) -> str:
        return f"forward_world_modeling_{self.step_length}_steps"
    
    def visual_prompt_path(self, image_root_dir) -> str:
        """
        Get the path for storing visual prompt images.

        Args:
            image_root_dir: Root directory for images.
            
        Returns:
            str: Path to the visual prompt directory (QA/images/[qa_type]).
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
        Check if a transition from frame_a to frame_b is valid.
        
        A transition is valid if it contains meaningful changes (1-5 state change sentences).
        
        Args:
            frame_a_id: Starting frame ID.
            frame_b_id: Ending frame ID.
            task_data: Task data containing scene graphs.
            
        Returns:
            bool: True if the transition is valid.
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
            key_frame_ids: List of frame IDs to consider.
            task_data: Task data containing scene graphs.
            
        Returns:
            Dict mapping each frame ID to a list of valid successor frame IDs.
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
            graph: Adjacency list of valid transitions.
            key_frame_ids: List of frame IDs.
            frame_to_index: Mapping from frame ID to index.
            
        Returns:
            numpy array where dp_table[k][i] is the count of paths of length (k+1) ending at frame i.
        """
        num_frames = len(key_frame_ids)
        dp_table = np.zeros((self.step_length, num_frames), dtype=np.int64)

        # Base case: each frame is a valid path of length 1
        dp_table[0, :] = 1

        print("Phase 2: Counting valid paths with Dynamic Programming...")
        
        # Fill the DP table layer by layer
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

    def _generate_qa_id_hash(self, task_name: str, qa_type: str, sequence: List[str]) -> str:
        """
        Generate a unique hash-based ID for a QA pair.
        
        Args:
            task_name: Name of the task.
            qa_type: Type of QA (e.g., 'forward_world_modeling_5_steps').
            sequence: Sequence of frame IDs.
            
        Returns:
            Unique ID string in format: {task_name}_{qa_type}_{hash}.
        """
        sequence_str = '_'.join(sequence)
        sequence_hash = hashlib.sha256(sequence_str.encode()).hexdigest()[:8]
        return f"{task_name}_{qa_type}_{sequence_hash}"

    
    def _translate_sequence_to_actions(self, task_data: TaskData, sequence: List[str]) -> str:
        """
        Translate a sequence of frames into numbered action descriptions.
        
        Args:
            task_data: Task data containing scene graphs.
            sequence: List of frame IDs in sequential order.
            
        Returns:
            Formatted string with numbered actions (e.g., "[Action 1] ...\n[Action 2] ...").
            
        Raises:
            ValueError: If no actions are found in the sequence.
        """
        action_descriptions = []
        for i in range(len(sequence) - 1):
            frame_a_id = sequence[i]
            frame_b_id = sequence[i+1]
            diff = task_data.scene_graph_reader.get_visible_full_diff(
                frame_a_id, frame_b_id, self.sensor_names, partial_diff=True
            )
            action_desc = self.translator.translate_diff(diff)
            action_descriptions.append(action_desc)

        if not action_descriptions:
            raise ValueError("No actions are performed.")
        
        # Format as numbered actions
        formatted_actions = []
        action_template = "[Action {i}] {action}"
        for i, desc in enumerate(action_descriptions):
            action = action_template.format(i=i+1, action=desc)
            formatted_actions.append(action)
            
        return "\n".join(formatted_actions)
    
    def _add_text_to_image(self, image_path: str, text: str, output_path: str) -> None:
        """
        Add a text label to an image and save it.
        
        Args:
            image_path: Path to the source image.
            text: Text to add to the image.
            output_path: Path where the labeled image will be saved.
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

    def generate(
            self, 
            task_data: TaskData, 
            num_to_sample: int=30
        ) -> List[QAPair]:
        """
        Generate forward world modeling QA pairs for a task.
        
        This method:
        1. Builds a graph of valid frame transitions (Phase 1)
        2. Counts all valid paths using dynamic programming (Phase 2)
        3. Samples unique paths using weighted random backtracking (Phase 3)
        4. Creates QA pairs from the sampled sequences (Phase 4)
        
        A valid sequence [f1, f2, ..., fk] has every consecutive pair (fi, f_{i+1})
        meeting the state change criteria (1-5 meaningful changes).

        Args:
            task_data: Task data containing scene graphs and images.
            num_to_sample: Number of unique sequences to sample.
            
        Returns:
            List of generated QA pairs.
        """
        # Set seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        key_frame_ids = sorted(task_data.key_frame_ids, key=int)
        num_frames = len(key_frame_ids)

        if num_frames < self.step_length:
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
        actual_num_to_sample = min(num_to_sample, total_paths)
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

        print(f"\nGenerated {len(qa_pairs)} multi-step forward world modeling QA pairs.")
        return qa_pairs

    def _create_ordering_qa_pair(
            self, 
            task_data: TaskData, 
            correct_sequence: List[str]
        ) -> QAPair:
        """
        Create a QA pair for the forward world modeling ordering task.
        
        Given a sequence of frames, this creates a question asking the user to order
        shuffled future states based on a sequence of actions.
        
        Args:
            task_data: Task data containing scene graphs and images.
            correct_sequence: List of frame IDs in the correct temporal order.
            
        Returns:
            A QAPair object, or None if the sequence is invalid.
        """
        assert len(correct_sequence) > 2, "Correct sequence must be at least 3 frames long"

        # Get the initial state image
        start_frame_id = correct_sequence[0]
        sensor_name = self.sensor_names[0]
        start_image_path = task_data.image_paths.get(start_frame_id, {}).get(sensor_name, None)
        if not start_image_path:
            return None
        
        # Generate action descriptions from the sequence
        action_descriptions = self._translate_sequence_to_actions(task_data, correct_sequence)
        
        # Get the next states (excluding the initial state)
        next_states = correct_sequence[1:]
        
        # Shuffle the next states for the ordering task
        shuffled_next_states = next_states[:]
        random.shuffle(shuffled_next_states)
        
        # Find the correct order (1-indexed positions)
        correct_order = []
        for original_frame in next_states:
            shuffled_position = shuffled_next_states.index(original_frame) + 1
            correct_order.append(shuffled_position)
        
        # Generate unique QA pair ID
        qa_id = self._generate_qa_id_hash(task_data.task_name, self.qa_type, correct_sequence)
        
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
            
            # Add labels to shuffled next state images
            for i, frame_id in enumerate(shuffled_next_states):
                next_state_image_path = task_data.image_paths[frame_id][sensor_name]
                label = f"Next State {i + 1}"
                next_state_output_path = output_dir / f"{qa_id}_next_state_{i + 1}.png"
                self._add_text_to_image(next_state_image_path, label, str(next_state_output_path))
                final_next_state_images.append(str(next_state_output_path))
        else:
            # Use original images without labels
            for frame_id in shuffled_next_states:
                image_path = task_data.image_paths[frame_id][sensor_name]
                final_next_state_images.append(image_path)
        
        # Generate the question
        question = multi_fwd_ordering_prompt.format(STATE_CHANGES=action_descriptions)
        
        # Prepare all images with relative paths
        all_images = [final_start_image] + final_next_state_images
        all_images = [str(Path(image_path).relative_to(image_root_dir)) for image_path in all_images]
        
        # Create the ground truth answer
        gt_answer = {
            "type": f"{self.qa_type}",
            "options": [],
            'correct_option': correct_order
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