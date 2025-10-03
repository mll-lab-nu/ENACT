# ENACT

ENACT is a benchmark that evaluates embodied cognition through world modeling from egocentric interaction. It is designed to be simple and have a scalable dataset.

## Project Structure

```
ENACT/
├── enact/                          # Main Python package
│   ├── core/                       # Core algorithms and managers
│   │   ├── segmentation.py        # Frame segmentation
│   │   ├── qa_generation.py       # QA pair generation
│   │   ├── evaluators.py          # Evaluation logic
│   │   ├── forward_dynamics_generator.py
│   │   └── inverse_dynamics_generator.py
│   ├── processors/                 # Batch processors
│   │   ├── segmentation_processor.py  # Segmentation processor (single/batch)
│   │   ├── evaluator_processor.py # Evaluation processor (single/batch)
│   │   └── qa_processor.py        # QA generation processor
│   └── utils/                      # Utility functions
│       ├── scene_graph_utils.py
│       ├── frame_seg_utils.py
│       ├── qa_gen_utils.py
│       ├── qa_prompt_template.py
│       └── state_change_translator.py
├── scripts/                        # Executable scripts
│   ├── enact/                     # Scripts for ENACT module
│   │   ├── run_segmentation.py    # Segmentation script (single/batch)
│   │   ├── run_qa_generation.py   # QA generation script
│   │   ├── run_eval.py            # Evaluation script (single/batch)
│   │   ├── run_batch_eval.py      # Legacy batch evaluation
│   │   └── evaluator_usage_example.py
│   └── helpers/                    # Helper scripts
│       └── (download_dataset.py, etc.)
├── data/                           # Data directory
├── setup.py                        # Package installation
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Installation

Install the package in development mode:

```bash
pip install -e .
```

This will make the `enact` package available for import and install console commands.

## Usage

### Running Scripts

After installation, you can run scripts directly from the command line:

```bash
# Run segmentation (supports both single task and batch mode)
python scripts/enact/run_segmentation.py [input_root] [output_root]

# Run QA generation
python scripts/enact/run_qa_generation.py [input_root] [raw_data_dir] [output_file]

# Run evaluation (supports both single file and batch mode)
python scripts/enact/run_eval.py [input_path]

# For detailed usage and options
python scripts/enact/run_segmentation.py --help
python scripts/enact/run_qa_generation.py --help
python scripts/enact/run_eval.py --help
```

Default paths:
- Segmentation: `data/replayed_activities` → `data/segmented_activities`
- QA Generation: Uses segmented and replayed activities
- Evaluation: `data/evaluation` (expects `enact_ordering_*.jsonl` files)

### Using as a Library

You can also import and use the modules in your own Python code:

```python
from enact.processors import SegmentationProcessor, EvaluatorProcessor
from enact.core.evaluators import OrderingEvaluator

# Segmentation - supports both single task and batch mode
seg_processor = SegmentationProcessor(input_root, output_root)
seg_processor.process_all_tasks()

# Evaluation - supports both single file and batch mode
eval_processor = EvaluatorProcessor(
    input_path="data/evaluation",
    segmented_data_dir="data/segmented_activities",
    raw_data_dir="data/replayed_activities",
    output_root="data/evaluation"
)
eval_processor.process_all_files()

# Or use core evaluator directly
evaluator = OrderingEvaluator(
    input_root_dir="data/segmented_activities",
    raw_data_dir="data/replayed_activities"
)
evaluator.evaluate("path/to/model_output.jsonl")
```

For detailed evaluation usage, see [EVALUATION_USAGE.md](EVALUATION_USAGE.md).

## Development

For development, install additional dependencies and run tests:

```bash
pip install -r requirements.txt
# Add testing commands here
```

## Adding Helper Scripts

Place helper scripts (like dataset download utilities) in `scripts/helpers/`:

```bash
scripts/helpers/download_dataset.py
scripts/helpers/preprocess_data.py
```

These scripts can access the `enact` module after installation.
