# SimulationCalibrator

An LLM agent that self-calibrates cosmological simulation parameters against a target
statistic. The agent iteratively proposes parameters for an [MP-Gadget](https://github.com/MP-Gadget/MP-Gadget)
simulation, runs the simulation, scores the resulting matter power spectrum P(k)
against a hidden ground-truth target with a mode-count-weighted χ², and updates its
estimate — aiming to reach an acceptable calibration in as few simulator queries (and
thus CPU-hours) as possible.

Each agent invocation is a fresh, stateless process: its only memory between trials is
what it writes to disk (`lab_notebook.md`, `trial_state.yaml`). A driver loop keeps
invoking fresh instances, one trial per invocation, until the agent reports
convergence or the CPU-hour budget runs out.

## Layout

```
src/                     Calibration engine and the agent's tools
  calibration_env.py       Writes paramfiles, submits the Slurm job, scores χ² (core engine)
  run_trial_daemon.py      Lock-protected submit-and-wait for one trial
  camb_proxy.py            Free CAMB (halofit) P(k) — a low-fidelity tool the agent may call
templates/               Scaffolding for a calibration run
  new_experiment.py        Generates a run directory from the templates below
  box_config/              The high-resolution box definition + MP-Gadget paramfile templates
  judge/                   Draws the hidden theta_fid and generates the target P(k)
  agent_workspace/         The agent's task prompt, lab notebook, and trial-state files
  code/                    The driver loop that invokes fresh agent instances
tests/                   Unit tests for the engine and the CAMB tool
```

## Requirements

- **MP-Gadget**, cloned and built separately. The engine imports `make_class_power`
  from its `tools/` directory and runs its `MP-GenIC` / `MP-Gadget` binaries. Point
  the `MPGADGET_ROOT` environment variable at your checkout (it defaults to a
  `sims/MP-Gadget` path that you will need to override).
- A **Slurm** cluster: trials are submitted with `sbatch` and polled with
  `squeue`/`sacct`. The account, partition, and resource settings live in
  `src/calibration_env.py` and `templates/box_config/hr_box_config.py.tmpl`, and are
  specific to the cluster they were written for — edit them for yours.
- Python with `numpy`, `camb`, and `pyyaml` (plus MP-Gadget's own CLASS dependency for
  `make_class_power`). The scripts assume a virtualenv at `.venv/`.

## Running a calibration

```bash
# 1. Scaffold a run directory from the templates
python3 templates/new_experiment.py --name my-run --job-prefix exp1

# then, inside the generated run directory:
# 2. Draw the hidden ground-truth parameters (no simulation; reproducible from the seed)
.venv/bin/python3 judge/generate_theta_fid.py
# 3. Run that ground truth once to produce the target P(k) (a real Slurm job)
.venv/bin/python3 judge/generate_target.py
# 4. Set budget_cpu_hours_cap in agent_workspace/trial_state.yaml
# 5. Launch the driver loop (inside tmux)
code/driver_llm-agent.sh
```

The ground-truth parameters are never exposed to the agent — only the resulting target
P(k). The agent's job is to recover them by physical reasoning and iterative
simulation. It is considered converged when a real trial's reduced χ² falls below 1
across the scored k-range.

## Tests

```bash
.venv/bin/python3 -m pytest tests/
```
