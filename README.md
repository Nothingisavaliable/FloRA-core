# FloRA Core

FloRA Core is the open-source research code for the FLoRA paper on scene-conditioned, multi-realization pouring. It covers task-realization representation, Global Motion generation, a Local closed-loop policy, robot embodiment evaluation, and experiment tracking.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install FloRA Core and the dependencies needed for model execution:

```bash
python -m pip install -e '.[models]'
```

Install optional dependencies as needed:

```bash
python -m pip install -e '.[simulation]'  # MuJoCo replay and point-cloud capture
python -m pip install -e '.[spacemouse]'  # SpaceMouse demonstration collection
python -m pip install -e '.[local]'       # PyTorch and Diffusers for Local DP3
python -m pip install -e '.[robotics]'    # MuJoCo, robosuite, and robot OSC adapters
python -m pip install -e '.[particles]'   # Taichi and MuJoCo particle-fluid simulation
python -m pip install -e '.[test]'        # pytest
```

To develop models and Local DP3, run robot simulation, and run the tests:

```bash
python -m pip install -e '.[models,local,robotics,test]'
```

LIBERO assets are resolved from the installed package. For a custom asset layout, pass `--libero-root` to the replay command.

After installing the test dependencies, run:

```bash
python -m pytest
```

## Quick Start

### Collect SpaceMouse demonstrations

Collect a pouring demonstration with a SpaceMouse in the MuJoCo viewer. The blue line shows the planned corridor, and the orange markers track the source outlet and target anchor. This example uses a bundled collection-design fixture.

```bash
python -m pip install -e '.[spacemouse,simulation]'
flora-collect-spacemouse \
  --collection-config tests/fixtures/protocol.json \
  --collection-design tests/fixtures/design.json \
  --collection-row-id PB_001 \
  --out-dir outputs/demonstrations
```

Follow the terminal prompts to record and save a demonstration. Use `--vendor-id` and `--product-id` if your SpaceMouse differs from the AEXT default. On macOS, use `mjpython -m flora_core.demonstrations.spacemouse` with the same arguments.

### Replay demonstrations and capture point clouds

Replay each saved demonstration to create a `dp3_episode.npz` with scene point clouds. Use a separate output directory for each demonstration.

```bash
for episode in outputs/demonstrations/spacemouse_*.npz; do
  flora-replay-demo "$episode" \
    --out-dir "outputs/replay/$(basename "$episode" .npz)" \
    --collect-point-cloud
done
```

An experiment configuration example is provided at `configs/experiment.example.json`. `flora_core.experiments` provides configuration loading and run recording; callers supply the training or evaluation loop.

## Modules

The module layout follows the paper's method:

```text
Task scene and realization
          │
          ├── Global Motion Predictor ── candidate remaining motion
          │                              │
          │                    Stage 1: OSC collision filtering
          │                              │ survivors
          │                    Stage 2: Tracking Cost ranking
          │                              │ selected realization
          └── Local DP3 ◀── Global guidance
                       │
                 OSC robot control
                       │
          re-predict and evaluate after a condition change
```

| Module | Description |
| --- | --- |
| `flora_core.data` | Scene data and model inputs |
| `flora_core.demonstrations` | Episode data and Local training windows |
| `flora_core.execution` | Global plan execution contracts |
| `flora_core.experiments` | Experiment configuration and run records |
| `flora_core.geometry` | Coordinate frames and rigid transforms |
| `flora_core.models` | Global Motion and Local DP3 models |
| `flora_core.particles` | Taichi particle-fluid and MuJoCo particle simulation backends |
| `flora_core.realization` | Task-realization representation and evaluation |
| `flora_core.robotics` | OSC shadow rollout and Global Motion embodiment evaluation |
| `flora_core.simulation` | MuJoCo replay and point-cloud capture |
| `flora_core.pipeline` | Candidate evaluation and plan selection |
