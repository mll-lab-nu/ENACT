# ENACT: Embodied Cognition through World Modeling from Egocentric Interaction

**ENACT** is a benchmark that evaluates embodied cognition through world modeling from egocentric interaction. It is designed to be simple and have a scalable dataset for evaluating forward and inverse dynamics in embodied AI systems.

The benchmark tests models on their ability to:
- **Forward World Modeling**: Predict the correct sequence of future states given a current state and a series of actions
- **Inverse World Modeling**: Infer the correct sequence of actions that led from an initial state to a sequence of observed future states

---

## Table of Contents

- [Environment Installation](#environment-installation)
- [Data Download](#data-download)
- [Data Evaluation](#data-evaluation)
- [Optional: Generate Data Yourself](#optional-generate-data-yourself)
  - [Pipeline Overview](#pipeline-overview)
  - [Stage 0: Collect Robot Data](#stage-0-optional-collect-robot-data--raw_hdf5)
  - [Stage 1: Replay HDF5](#stage-1-optional-replay-hdf5--replayed_activities)
  - [Stage 1.5: Extract Frames](#stage-15-extract-frames-from-videos--replayed_activitiesexternal_sensor1)
  - [Stage 2: Segment Activities](#stage-2-segment-activities--segmented_activities)
  - [Stage 3: Generate QA](#stage-3-generate-qa-tasks--qaenact_orderingjsonl)
- [Simulator Installation](#simulator-installation-optional)

---

## Environment Installation

> **⚠️ IMPORTANT:** If you plan to use the **BEHAVIOR-1K simulator** for data generation (replaying HDF5 files), **skip steps 2** and jump directly to [Simulator Installation](#simulator-installation-optional) section below. The simulator setup will create its own conda environment with all required dependencies. After installing with the simulator env, get back to **step 3**.

### 1. Clone the Repository

```bash
git clone git@github.com:QinengWang-Aiden/ENACT.git
cd ENACT/
```

### 2. Create Conda Environment (Skip if using simulator)

Create a new conda environment named `enact` with Python 3.10:

```bash
conda create -n enact python=3.10 -y
conda activate enact
```

### 3. Install the ENACT Package

Install the package in editable mode:

```bash
pip install -e .
# Verify installation
enact --help
```

---

## Data Download

By default, ENACT downloads the **ENACT QA dataset** which contains question-answer pairs with images for VLMs evaluation. You can optionally download additional datasets like HDF5 files, replayed activities, and segmented activities.

### Quick Start: Download ENACT QA Dataset

```bash
# Download only ENACT QA
python scripts/helpers/download_dataset.py
# Download ALL datasets
python scripts/helpers/download_dataset.py --all
```

This downloads the QA dataset (approximately 17 GB) to `data/QA/` by default.

<details>
<summary>Complete options</summary>

```bash
# Download only ENACT QA dataset (default)
python scripts/helpers/download_dataset.py --output-dir ./data

# Skip ENACT QA dataset if you don't need it
python scripts/helpers/download_dataset.py --no-enact

# Download HDF5 dataset (raw simulation recordings)
python scripts/helpers/download_dataset.py --hdf5

# Download replayed activities (extracted scene graphs and frames)
python scripts/helpers/download_dataset.py --replayed

# Download segmented activities (segmented scene graphs)
python scripts/helpers/download_dataset.py --segmented
```

**Dataset Descriptions:**
- **ENACT QA** (default, ~17 GB): Contains `enact_ordering.jsonl` with 8972 QA pairs and associated images for evaluation
- **HDF5** (Optional): Raw simulation recordings from BEHAVIOR-1K simulator
- **Replayed Activities** (Optional): Scene graphs and extracted frames from replayed HDF5 files
- **Segmented Activities** (Optional): Segmented scene graphs with action boundaries identified

</details>

### Understanding the Downloaded Data Structure

After downloading, your `data/` directory will contain:

```
data/
├── QA/                              # ENACT QA dataset
│   ├── enact_ordering.jsonl        # 8972 QA pairs
│   └── images/                      # Associated images
│       ├── forward_world_modeling_ordering_3_steps/
│       ├── forward_world_modeling_ordering_4_steps/
│       ├── ...
│       ├── inverse_world_modeling_ordering_3_steps/
│       └── ...
├── raw_hdf5/                        # (Optional) Raw simulation data
├── replayed_activities/             # (Optional) Extracted scene graphs
└── segmented_activities/            # (Optional) Segmented frames
```

---

## Data Evaluation

### Understanding the Dataset Format

Each line in `enact_ordering.jsonl` contains a QA instance with the following structure.

**Key Fields:**
- `id`: Unique identifier for this QA instance
- `type`: Question type (forward/inverse world modeling with N steps)
- `images`: List of image paths - first is current state, rest are shuffled future states
- `question`: Full prompt with task description and actions
- `gt_answer`: Ground truth ordering (e.g., `[2, 1]` means the correct order is image 2 then image 1)

<details>
<summary>Example input format</summary>

```json
{
  "id": "task_name_type_hash",
  "type": "forward_world_modeling_ordering_3_steps",
  "task_name": "assembling_gift_baskets_1749468508582193",
  "key_frame_ids": ["16084", "18290", "18501"],
  "images": [
    "QA/images/.../cur_state.png",
    "QA/images/.../next_state_1.png",
    "QA/images/.../next_state_2.png"
  ],
  "question": "You are a capable agent...",
  "options": [],
  "gt_answer": [2, 1]
}
```

</details>

### Preparing Your Model Output

Your model should generate a JSONL file where each line contains the original fields plus an `answer` field.

**Requirements:**
- All fields except `answer` must match the input `enact_ordering.jsonl`
- `answer` should be a **string** containing a parsable list (e.g., `"[2, 1]"` instead of `[2, 1]`)
- **Recommended naming:** `enact_ordering_{model_name}.jsonl`

<details>
<summary>Example model output format</summary>

```json
{
  "id": "task_name_type_hash",
  "type": "forward_world_modeling_ordering_3_steps",
  "task_name": "assembling_gift_baskets_1749468508582193",
  "key_frame_ids": ["16084", "18290", "18501"],
  "gt_answer": [2, 1],
  "answer": "[2, 1]"
}
```

</details>

### Running Evaluation

```bash
# single file evaluation
enact eval your_model_output.jsonl
# batch file evaluation
# the evaluator will look for files matching pattern "enact_ordering_*.jsonl"
enact eval model_outputs_directory/
```

<details>
<summary>Complete version with all options</summary>

```bash
# Specify custom data paths
enact eval your_model_output.jsonl \
  --segmented-data data/segmented_activities \
  --raw-data data/replayed_activities \
  --output-root data/evaluation

# Enable detailed wrong case output
enact eval your_model_output.jsonl --analyze-wrong-cases

# Preview what would be evaluated without running
enact eval your_model_output.jsonl --dry-run
```

**Arguments:**
- `input_path`: Path to JSONL file or directory containing JSONL files
- `--segmented-data`: Path to segmented activities (default: `data/segmented_activities`)
- `--raw-data`: Path to replayed activities (default: `data/replayed_activities`)
- `--output-root`: Where to save evaluation results (default: `data/evaluation`)
- `--analyze-wrong-cases`: Generate detailed signatures for incorrect predictions
- `--dry-run`: Show what would be evaluated without actually processing

</details>

### Understanding Evaluation Results

After evaluation, results are saved to the output directory (default: `data/evaluation/`):

```
data/evaluation/
├── batch_evaluation_summary.json   # Overall summary across all models
├── meta_performance/               # Summary metrics per model
│   └── enact_ordering_modelname.json
├── detailed_eval/                  # Per-sample detailed results (JSONL)
│   └── enact_ordering_modelname.jsonl
└── signatures/                     # (If --analyze-wrong-cases enabled, JSONL)
    └── enact_ordering_modelname.jsonl
```

**Note:** The evaluator extracts model name from the input filename. For example:
- Input: `enact_ordering_gpt-4.jsonl` → Output files: `enact_ordering_gpt-4.json` / `.jsonl`
- Input: `my_model_predictions.jsonl` → Model name: `my_model_predictions`

#### Meta Performance File

Contains aggregated metrics with overall and per-task-type breakdowns.

**Key Metrics:**
- `model_name`: Name of the model being evaluated (extracted from filename)
- `overall_performance.overall`: Overall performance across all question types
  - `count`: Total number of QA instances evaluated
  - `task_accuracy`: Percentage of correctly ordered sequences (exact match)
  - `pairwise_accuracy`: Percentage of correct pairwise orderings
- `forward_world_modeling` / `inverse_world_modeling`: Breakdown by dynamics type

<details>
<summary>Example JSON output</summary>

```json
{
  "model_name": "human",
  "overall_performance": {
    "overall": {
      "count": 8972,
      "task_accuracy": 0.8859786000891663,
      "pairwise_accuracy": 0.9492396096497747
    },
    "forward_world_modeling": {
      "count": 4486,
      "task_accuracy": 0.879402585822559,
      "pairwise_accuracy": 0.9481513916311064
    },
    "inverse_world_modeling": {
      "count": 4486,
      "task_accuracy": 0.8925546143557735,
      "pairwise_accuracy": 0.9503278276684429
    }
  }
}
```

</details>

#### Detailed Evaluation File

Contains per-sample results with individual predictions and correctness (JSONL format, one JSON object per line).

**Key Fields:**
- `eval_metrics`: Multiple accuracy measures
  - `exact_match`: Whether the full sequence matches exactly
  - `semantic_match`: Whether the meaning matches (allows reordering of simultaneous events)
  - `task_accuracy`: Task-level correctness (same as exact_match)
  - `pairwise_accuracy`: Percentage of correct pairwise orderings (partial credit)
- `ground_truth`: Correct ordering
- `model_answer`: Model's predicted ordering
- `raw_answer`: Raw string output from the model
- `wrong_case_analysis`: Detailed breakdown (always included, even for correct answers)

<details>
<summary>Example JSONL entry</summary>

```json
{
  "id": "assembling_gift_baskets_1749468508582193_forward_dynamics_ordering_3_steps_5dc7cfd5",
  "task_name": "assembling_gift_baskets_1749468508582193",
  "type": "forward_dynamics_ordering_3_steps",
  "eval_metrics": {
    "exact_match": false,
    "semantic_match": false,
    "task_accuracy": false,
    "pairwise_accuracy": 0.5
  },
  "ground_truth": [2, 1],
  "model_answer": [1, 2],
  "raw_answer": "[1, 2]",
  "wrong_case_analysis": {
    "id": "...",
    "type": "...",
    "key_frame_ids": ["16084", "18290", "18501"],
    "gt_answer": [2, 1],
    "parsed_answer": [1, 2],
    "correct_signatures": [["edge_add_..."], ["edge_remove_..."]],
    "input_signatures": [["edge_remove_...", "edge_add_..."], ["edge_add_..."]],
    "correct_natural_language": ["Action 1 description", "Action 2 description"],
    "input_natural_language": ["Wrong action 1", "Wrong action 2"]
  }
}
```

</details>

#### Wrong Case Signatures (Optional)

When `--analyze-wrong-cases` is enabled, generates detailed analysis with action signatures (JSONL format, one JSON object per line).

**Signature Analysis Fields:**
- `correct_signatures`: The actual state changes at each step (as edge operations)
- `input_signatures`: The state changes predicted by the model
- `correct_natural_language`: Human-readable description of correct transitions
- `input_natural_language`: Human-readable description of model's predictions
- `equal_length`: Whether model output has the correct number of steps

This file helps you understand **why** the model made mistakes by comparing the predicted state transitions with the ground truth.

<details>
<summary>Example JSONL entry</summary>

```json
{
  "id": "assembling_gift_baskets_1749468508582193_forward_dynamics_ordering_3_steps_5dc7cfd5",
  "type": "forward_dynamics_ordering_3_steps",
  "task_name": "assembling_gift_baskets_1749468508582193",
  "key_frame_ids": ["16084", "18290", "18501"],
  "gt_answer": [2, 1],
  "parsed_answer": [1, 2],
  "raw_answer": "[1, 2]",
  "eval_metrics": {
    "exact_match": false,
    "semantic_match": false,
    "task_accuracy": false,
    "pairwise_accuracy": 0.5
  },
  "equal_length": true,
  "correct_signatures": [
    ["edge_add_the robot r1_the butter cookie_LeftGrasping"],
    ["edge_remove_the butter cookie_the coffee table_OnTop"]
  ],
  "input_signatures": [
    ["edge_remove_the butter cookie_the coffee table_OnTop", "edge_add_the robot r1_the butter cookie_LeftGrasping"],
    ["edge_add_the butter cookie_the coffee table_OnTop"]
  ],
  "correct_natural_language": [
    "The robot r1 changes to be using the left gripper to grasp the butter cookie.",
    "The butter cookie stopped being on top of and touching the coffee table."
  ],
  "input_natural_language": [
    "The robot r1 changes to be using the left gripper to grasp the butter cookie. The butter cookie is no longer on top of and touching the coffee table.",
    "The butter cookie transitions to be on top of and touching the coffee table."
  ]
}
```

</details>

#### Batch Evaluation Summary (When Evaluating Multiple Models)

When evaluating a directory with multiple model outputs, a `batch_evaluation_summary.json` is created. This provides a quick comparison across all evaluated models.

<details>
<summary>Example JSON output</summary>

```json
{
  "total_processed": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    {
      "model_name": "gpt-5-mini-2025-08-07",
      "status": "success",
      "overall_stats": {
        "count": 8972,
        "task_accuracy": 0.3695,
        "pairwise_accuracy": 0.6474
      }
    },
    {
      "model_name": "human",
      "status": "success",
      "overall_stats": {
        "count": 8972,
        "task_accuracy": 0.8860,
        "pairwise_accuracy": 0.9492
      }
    }
  ]
}
```

</details>

### Example Evaluation Workflow

```bash
# 1. Download the ENACT QA dataset
python scripts/helpers/download_dataset.py

# 2. Run your model on data/QA/enact_ordering.jsonl to generate predictions
# Your model should output: enact_ordering_mymodel.jsonl

# 3. Evaluate your predictions
enact eval enact_ordering_mymodel.jsonl --analyze-wrong-cases

# 4. Check results
cat data/evaluation/meta_performance/enact_ordering_mymodel.json

# 5. For batch evaluation of multiple models
enact eval model_outputs_directory/ --analyze-wrong-cases
cat data/evaluation/batch_evaluation_summary.json
```

---

## Optional: Generate Data Yourself

The ENACT dataset generation follows a multi-stage pipeline. **You can start from any stage** as we provide official intermediate datasets for each stage. Only **Stage 1** (replaying HDF5 files) requires the BEHAVIOR-1K simulator.

### Pipeline Overview

```
Stage 0 (Optional): Collect Robot Data   → raw_hdf5/
                                            ↓ (requires simulator)
Stage 1 (Optional): Replay HDF5          → replayed_activities/ (mp4 + scene_graph)
                                            ↓
Stage 1.5:          Extract Frames        → replayed_activities/*/external_sensor1/
                                            ↓
Stage 2:            Segment Activities    → segmented_activities/ (key frames only)
                                            ↓
Stage 3:            Generate QA           → QA/enact_ordering.jsonl
```

**Official Data Sources:**
- **raw_hdf5**: [Google Drive (Ours)](https://drive.google.com/file/d/1B3YTxlV5V7T8UqkY1V4ReF5jkuSu2qrs/view?usp=sharing) or [Behavior HuggingFace (29 tasks, 200 trajectories each)](https://huggingface.co/datasets/behavior-1k/2025-challenge-rawdata)
- **replayed_activities**: [Google Drive](https://drive.google.com/file/d/19rkSTPZmm2eWfuro8juv3acimELgD-xb/view?usp=sharing)
- **segmented_activities**: [Google Drive](https://drive.google.com/file/d/1sPS7Lxw-FBPcWJbh7hOaD-22OIet_QrR/view?usp=sharing)
- **QA dataset**: [HuggingFace (default)](https://huggingface.co/datasets/Inevitablevalor/ENACT)

---

### Stage 0 (Optional): Collect Robot Data → `raw_hdf5/`

**⚠️ Coming Soon:** Tutorial for collecting your own robot trajectories using BEHAVIOR-1K simulator.

**Use Official Data Instead:**
- **Option 1** - Our curated dataset (subset):
  ```bash
  python scripts/helpers/download_dataset.py --hdf5
  ```
- **Option 2** - Full HuggingFace dataset (29 tasks × 200 trajectories):
  - Visit: https://huggingface.co/datasets/behavior-1k/2025-challenge-rawdata
  - This is all available hdf5 datasets used in BEHAVIOR Challenge.

**Output:** `data/raw_hdf5/` containing HDF5 simulation recordings

---

### Stage 1 (Optional): Replay HDF5 → `replayed_activities/`

**⚠️ Requires BEHAVIOR-1K Simulator** - See [Simulator Installation](#simulator-installation-optional) for setup.

This stage replays HDF5 files in the simulator to extract:
- Scene graphs (object relationships and states at each timestep)
- MP4 video (egocentric camera view)

**Run Replay (Single File):**
```bash
# After installing simulator
python scripts/helpers/replay_hdf5.py --file data/raw_hdf5/task_name.hdf5 --output_dir data/replayed_activities
```

**Run Replay (Batch Mode - All Files):**
```bash
# Processes all HDF5 files in data/raw_hdf5/
bash scripts/helpers/batch_replay_hdf5.sh
```

**Or Download Official Replayed Data:**
```bash
python scripts/helpers/download_dataset.py --replayed
```

**Output Structure:** `data/replayed_activities/`
```
replayed_activities/
├── assembling_gift_baskets_1749468508582193/
│   ├── external_sensor1.mp4       # Egocentric video
│   └── scene_graph_0.json         # Scene graph data
└── bringing_water_1750844141719178/
    ├── external_sensor1.mp4
    └── scene_graph_0.json
```

---

### Stage 1.5: Extract Frames from Videos → `replayed_activities/*/external_sensor1/`

**No simulator required.** Extract PNG frames from the MP4 videos produced in Stage 1. **This step is required before segmentation.**

**Input:** `data/replayed_activities/` with MP4 files

**Extract Frames (Single Task):**
```bash
python scripts/helpers/frame_extraction.py --task_folder data/replayed_activities/assembling_gift_baskets_1749468508582193
```

**Extract Frames (Batch Mode - All Tasks):**
```bash
python scripts/helpers/frame_extraction.py --task_folder data/replayed_activities
```

**Skip Already Processed:**
```bash
python scripts/helpers/frame_extraction.py --task_folder data/replayed_activities --skip_existing
```

**Output:** Frames are extracted into `external_sensor1/` subfolder in each task directory:
```
replayed_activities/
├── assembling_gift_baskets_1749468508582193/
│   ├── external_sensor1.mp4
│   ├── scene_graph_0.json
│   └── external_sensor1/              # New: extracted frames
│       ├── 00001.png
│       ├── 00002.png
│       └── ...
└── bringing_water_1750844141719178/
    └── ...
```

---

### Stage 2: Segment Activities → `segmented_activities/`

**No simulator required.** This stage processes scene graphs to identify key frames where significant state changes occur (action boundaries), then copies the corresponding frames.

**Input:** 
- `data/replayed_activities/` with extracted frames (from Stage 1.5)
- Scene graph JSON files

**Run Segmentation:**
```bash
# Basic usage (uses default paths)
enact segment

# Custom paths
enact segment data/replayed_activities data/segmented_activities

# Preview before processing
enact segment --dry-run
```

**Or Download Official Segmented Data:**
```bash
python scripts/helpers/download_dataset.py --segmented
```

**Output Structure:** `data/segmented_activities/`
```
segmented_activities/
├── assembling_gift_baskets_1749468508582193/
│   ├── external_sensor1/              # Segmented key frames
│   │   ├── 00059.png
│   │   ├── 00705.png
│   │   ├── 00916.png
│   │   └── ...                        # 53 key frames total
│   └── segmented_scene_graph_0.json   # Scene graph with only key frames
├── canning_food_1751278778230696/
│   ├── external_sensor1/              # 78 key frames
│   │   └── ...
│   └── segmented_scene_graph_0.json
└── bringing_water_1750844141719178/
    ├── external_sensor1/              # 15 key frames
    │   └── ...
    └── segmented_scene_graph_0.json
```

**Note:** Each task typically has 15-80 segmented frames representing key action boundaries. For example, `canning_food` has 78 segmented frames, which can generate over 0.5 billion possible 10-step ordering questions.

---

### Stage 3: Generate QA Tasks → `QA/enact_ordering.jsonl`

**No simulator required.** This stage samples state transitions from segmented data to create forward and inverse world modeling questions.

**Input:** 
- `data/segmented_activities/` (from Stage 2 or downloaded)
- `data/replayed_activities/` (for extracting images)

**Run QA Generation:**
```bash
# Basic usage (uses default paths)
enact qa

# Custom paths
enact qa data/segmented_activities data/replayed_activities data/QA/enact_ordering.jsonl

# Control sampling
enact qa --seed 42 --num-to-sample 10

# Preview before generating
enact qa --dry-run
```

**Or Download Official QA Dataset:**
```bash
python scripts/helpers/download_dataset.py  # Downloads QA by default
```

**Output:**
- `data/QA/enact_ordering.jsonl` - 8,972 QA pairs (in our paper's version)
- `data/QA/images/` - Organized by question type

**Data Generation Scale:**
For example, a task like `Canning Food` with 78 segmented frames can generate **over 0.5 billion** possible 10-step ordering questions. Our sampling strategy ensures diverse and challenging questions while maintaining computational feasibility.

<details>
<summary>Example QA entry structure</summary>

Each generated QA instance includes:
- **Question prompt**: Instructions for the model
- **Images**: Current state + shuffled future state images
- **Actions**: Ordered list of state transitions
- **Ground truth**: Correct ordering of future states

See [Data Evaluation](#data-evaluation) section for detailed format.

</details>

---

### Complete Pipeline Examples

**Example 1: Start from raw HDF5 (requires simulator)**
```bash
# 1. Install simulator (see Simulator Installation section)
# 2. Download HDF5 files
python scripts/helpers/download_dataset.py --hdf5
# 3. Replay HDF5 in simulator (batch mode)
bash scripts/helpers/batch_replay_hdf5.sh
# Or single file:
# python scripts/helpers/replay_hdf5.py --file data/raw_hdf5/task.hdf5 --output_dir data/replayed_activities
# 4. Extract frames from videos
python scripts/helpers/frame_extraction.py --task_folder data/replayed_activities
# 5. Segment activities
enact segment
# 6. Generate QA
enact qa --seed 42
```

**Example 2: Start from replayed activities (no simulator needed)**
```bash
# 1. Download replayed activities
python scripts/helpers/download_dataset.py --replayed
# 2. Extract frames from videos
python scripts/helpers/frame_extraction.py --task_folder data/replayed_activities
# 3. Segment activities
enact segment
# 4. Generate QA
enact qa --seed 42
```

**Example 3: Start from segmented activities (no simulator needed)**
```bash
# 1. Download segmented activities and replayed activities (for images)
python scripts/helpers/download_dataset.py --segmented --replayed
# 2. Generate QA
enact qa --seed 42
```

**Example 4: Only evaluate on official QA dataset (no generation)**
```bash
# 1. Download QA dataset (default)
python scripts/helpers/download_dataset.py
# 2. Run your model and evaluate
enact eval your_model_output.jsonl
```

---

## Simulator Installation (Optional)

**Only required if you want to replay HDF5 files (Stage 1).** The BEHAVIOR-1K simulator setup will create its own conda environment with all dependencies including OmniGibson, BDDL, and datasets.

> **⚠️ Important:** If you already created an `enact` conda environment following the earlier steps, but you want to use the simulator later, you may delete your old env and install with the simulator installation script.

### Setup Steps

**1. Initialize BEHAVIOR-1K submodule**
```bash
cd ENACT/
git submodule update --init --recursive
```

**2. Run BEHAVIOR-1K setup script**
```bash
cd BEHAVIOR-1K/
./setup.sh --new-env --omnigibson --bddl --joylo --dataset
```

This command will:
- Create a new conda environment
- Install OmniGibson simulator
- Install BDDL (Behavior Domain Definition Language)
- Download necessary datasets for simulation

**Setup time:** ~30-60 minutes depending on your internet connection and hardware.

### Verify Installation

After setup completes, verify the installation:

**Test 1: Launch Isaac Sim**
```bash
conda activate enact
isaacsim
```
This should open the Isaac Sim GUI. Close it after confirming it launches.

**Test 2: Run robot control example**
```bash
python OmniGibson/omnigibson/examples/robots/robot_control_example.py
```
This should run a simulation with robot control.

### Return to ENACT Environment

After verifying simulator installation, return to the ENACT root directory:

```bash
cd ..  
conda activate enact  
pip install -e .  
```

Now you can proceed with [Stage 1: Replay HDF5](#stage-1-optional-replay-hdf5--replayed_activities) to replay HDF5 files.

---

## Additional Commands and Help

### Get Help

```bash
# General help
enact --help

# Help for specific subcommands
enact segment --help
enact qa --help
enact eval --help
```

### Using as a Python Library

You can also import and use ENACT modules in your own Python code:

```python
from enact.processors import SegmentationProcessor, EvaluatorProcessor
from enact.core.evaluators import OrderingEvaluator

# Segmentation
seg_processor = SegmentationProcessor(
    input_root="data/replayed_activities",
    output_root="data/segmented_activities"
)
seg_processor.process_all_tasks()

# Evaluation
eval_processor = EvaluatorProcessor(
    input_path="model_output.jsonl",
    segmented_data_dir="data/segmented_activities",
    raw_data_dir="data/replayed_activities",
    output_root="data/evaluation",
    analyze_wrong_cases=True
)
eval_processor.process_all_files()
```

---

## Citation

If you use ENACT in your research, please cite:

```bibtex
@article{enact2025,
  title={ENACT: Embodied Cognition through World Modeling from Egocentric Interaction},
  author={ENACT Team},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

ENACT builds upon the BEHAVIOR simulator.
