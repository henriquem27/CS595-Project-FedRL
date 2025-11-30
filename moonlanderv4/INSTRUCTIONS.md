# Instructions - How to Run moonlanderv4

## Prerequisites

install docker on your system.

---

## Run Experiments

### option 1: using run script

```bash
cd ~/Desktop/CS595-Project-FedRL/moonlanderv4
chmod +x run.sh
./run.sh
```

### option 2: manual docker commands

```bash
cd ~/Desktop/CS595-Project-FedRL/moonlanderv4
docker build -t fedrl .
docker run -v $(pwd)/logs:/app/logs -v $(pwd)/plots:/app/plots fedrl
```

**This will run:**
- Standard FL with gradual weight adjustment (alpha=0.3)
- DP-FL with 4 sensitivity values (15, 10, 5, 1)
- Single agent baseline

**Expected time:** 2-8 hours (depends on CPU)

---

## Results

after the container finishes:
- **logs/** - training data and model weights
- **plots/** - generated visualizations
- **execution_times.txt** - timing information

plots are generated automatically at the end of the experiment.

---

## View Results

```bash
cat execution_times.txt
ls plots/
ls logs/
```

---

## Quick Test (Faster)

Edit `experiment.py` lines 65-67:
```python
NUM_ROUNDS = 5        # instead of 100
LOCAL_STEPS = 5000    # instead of 12500
```

Then run:
```bash
python experiment.py
```

**Expected time:** 30-60 minutes

---

## Individual Experiments

### Run only standard FL:
```bash
python fl_moon.py
```

### Run only DP-FL:
```bash
python dp_moon.py
```

### Run only single agent:
```bash
python single_moon.py
```

---

## Troubleshooting

### Out of memory:
Edit `fl_moon.py` and `dp_moon.py` line 45:
```python
N_ENVS = 16  # instead of 32
```

### Too many file descriptors:
```bash
ulimit -n 4096
```

### Import errors:
```bash
cd ~/Desktop/CS595-Project-FedRL/moonlanderv4
python -c "from helpers import gradual_weight_update; print('OK')"
```

---

## Output Structure

```
moonlanderv4/
├── logs/
│   ├── fl_run/
│   │   ├── weights/
│   │   │   ├── round_0/
│   │   │   │   ├── Client_1_Moon.pt
│   │   │   │   └── Global_Model.pt
│   │   │   └── round_1/ ...
│   │   ├── metrics/
│   │   │   ├── Client_1_Moon_metrics.csv
│   │   │   └── ...
│   │   └── global_model_final.zip
│   ├── dp_fl_sens15.0_eps30.0/
│   ├── dp_fl_sens10.0_eps30.0/
│   ├── dp_fl_sens5.0_eps30.0/
│   ├── dp_fl_sens1.0_eps30.0/
│   └── single_moon/
├── plots/
│   ├── fl_run_learning_curves.png
│   ├── all_experiments_comparison.png
│   └── per_client_comparison.png
└── execution_times.txt
```

---

## That's It!

Just run `python experiment.py` and wait for completion.
