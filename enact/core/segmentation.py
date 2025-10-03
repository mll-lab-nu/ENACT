#!/usr/bin/env python3
"""
Frame Segmentation Manager

Extensible framework for frame segmentation based on scene graph changes.
"""

import math
import sys
import os
import json
from collections import defaultdict

try:
    from enact.utils.scene_graph_utils import SceneGraphReader
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class FrameSegmentManager:
    """
    Manager for identifying key frames in video sequences based on scene graph changes.
    
    Uses cosine similarity to detect significant state transitions between frames.
    """
    
    def __init__(self, scene_graph_path: str):
        """
        Initialize the frame segmentation manager.
        
        Args:
            scene_graph_path: Path to the scene graph JSON file
        """
        self.scene_graph_path = scene_graph_path
        self.reader = SceneGraphReader(scene_graph_path, filter_transients=True)
        self.skip_first_frames_num = 10
        self.frame_ids = self.reader.get_available_frame_ids()[self.skip_first_frames_num:]
        self.extracted_frames = []
        self.working_camera = 'external_sensor1'
    
    def extract_changes(self, method: str = "cosine_similarity") -> dict:
        """
        Extract significant frame changes from the video sequence.
        
        Args:
            method: Segmentation method to use (default: "cosine_similarity")
            
        Returns:
            Dictionary mapping frame IDs to their changes (diffs)
        """
        if method == "cosine_similarity":
            return self._extract_cosine_similarity_changes()
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
        
    def _extract_features(self, scene_graph):
        """
        Extract bag-of-features representation from a scene graph.
        
        Creates feature strings from node and edge states for similarity comparison.
        """
        features = defaultdict(int)
        
        for node in scene_graph.get("nodes", []):
            node_name = node.get("name", "")
            for state in node.get("states", []):
                feature = f"node:{node_name}:{state}"
                features[feature] += 1
        
        for edge in scene_graph.get("edges", []):
            from_node = edge.get("from", "")
            to_node = edge.get("to", "")
            for state in edge.get("states", []):
                feature = f"edge:{from_node}->{to_node}:{state}"
                features[feature] += 1
        
        return features
    
    def _cosine_similarity(self, scene_graph1, scene_graph2):
        """
        Compute cosine similarity between two scene graphs.
        
        Converts scene graphs to feature vectors and computes their similarity.
        Returns 1.0 for identical graphs, 0.0 for completely different graphs.
        """
        features1 = self._extract_features(scene_graph1)
        features2 = self._extract_features(scene_graph2)

        vocabulary = set(features1.keys()) | set(features2.keys())

        vec1 = [features1.get(feature, 0) for feature in vocabulary]
        vec2 = [features2.get(feature, 0) for feature in vocabulary]

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
        magnitude2 = math.sqrt(sum(b ** 2 for b in vec2))

        if magnitude1 * magnitude2 == 0:
            return 1.0
        return dot_product / (magnitude1 * magnitude2)
    
    def _has_added_object(self, diff):
        """
        Check if a new object was added to the scene.
        
        Returns True if any added node has a parent attribute.
        """
        add_dict = diff['add']['nodes']
        for obj_dict in add_dict:
            if hasattr(obj_dict, 'parent'):
                return True
        return False
    
    def _extract_cosine_similarity_changes(self) -> dict:
        """
        Extract key frames using cosine similarity-based change detection.
        
        Identifies frames where significant scene changes occur by comparing
        cosine similarity between consecutive frames against a threshold.
        Implements temporal spacing to avoid capturing too many frames.
        """
        SKIPPING_FRAMES = 50
        SKIPPING_SAVED_INTERVAL = 10
        SKIPPING_ADDED_OBJECT_INTERVAL = 40
        STATE_THRESHOLD = 0.98
        TEMPORAL_THRESHOLD = 200

        changes = {}
        prev_frame = self.frame_ids[SKIPPING_FRAMES-1]
        prev_frame_number = int(prev_frame)
        prev_scene_graph = self.reader.get_scene_graph(prev_frame)
        changes[prev_frame] = {
            "type": "full",
            "nodes": prev_scene_graph['nodes'],
            "edges": prev_scene_graph['edges']
        }

        # State for postponed candidate tracking
        candidate_frame = None
        candidate_diff = None
        min_save_frame_number = None

        for i in range(SKIPPING_FRAMES, len(self.frame_ids) - 1):
            if i == SKIPPING_FRAMES:
                continue
            current_frame = self.frame_ids[i]
            current_frame_number = int(current_frame)
            current_scene_graph = self.reader.get_scene_graph(current_frame)

            similarity = self._cosine_similarity(prev_scene_graph, current_scene_graph)
            if similarity < STATE_THRESHOLD:
                if current_frame_number - prev_frame_number > TEMPORAL_THRESHOLD:
                    diff = self.reader.get_diff(prev_frame, current_frame)
                    assert diff['type'] != 'empty', f"Diff type is empty for frame {current_frame}"
                    
                    if candidate_frame is not None and current_frame_number >= min_save_frame_number:
                        # Save the current frame and update reference
                        changes[current_frame] = diff
                        changes[current_frame]['type'] = 'diff'
                        prev_frame = current_frame
                        prev_frame_number = current_frame_number
                        prev_scene_graph = current_scene_graph
                        
                        candidate_frame = None
                        candidate_diff = None
                        min_save_frame_number = None
                        
                    elif candidate_frame is None:
                        # Set as candidate for postponed saving
                        candidate_frame = current_frame
                        candidate_diff = diff
                        if self._has_added_object(diff):
                            min_save_frame_number = current_frame_number + SKIPPING_ADDED_OBJECT_INTERVAL
                            print(f"Skipping {SKIPPING_ADDED_OBJECT_INTERVAL} frames after an added object")
                        else:
                            min_save_frame_number = current_frame_number + SKIPPING_SAVED_INTERVAL

        # Save any remaining candidate at the end of the sequence
        if candidate_frame is not None:
            changes[candidate_frame] = candidate_diff
            changes[candidate_frame]['type'] = 'diff'

        self.extracted_frames = list(changes.keys())
        return changes


    def save_changes(self, changes: dict, output_path: str):
        """
        Save extracted frame changes to a JSON file.
        
        Args:
            changes: Dictionary of frame changes to save
            output_path: Path where the JSON file will be written
        """
        with open(output_path, 'w') as f:
            json.dump(changes, f, indent=2)
        
        print(f"Found {len(changes)} frames with changes")
        print(f"Saved to {output_path}")