# Topic 14 runbook

Environment: Python 3.10+, PyTorch >=2.2, NumPy >=1.24. No HF/Ray/distributed stack.

## 0. Enter and install

```bash
cd 14_powerlaw_persistent_head
pip install -r requirements.txt
```

## 1. CPU checks

```bash
python -m pytest -q
python audit_schedule.py --profile pilot
python audit_schedule.py --profile full
```

## 2. Smoke

```bash
bash run_gate.sh smoke 0
```

Never interpret smoke.

## 3. Pilot

```bash
bash run_gate.sh pilot 0
```

Pilot is signal-only. Do not keep/kill from one seed.

## 4. Full locked confirmation

```bash
bash run_gate.sh full 0,1,2,3,4
```

On a 4-GPU node, each seed uses four independent single-GPU arms in parallel. Seeds are run sequentially. Set `RESUME=1` after interruption.

## 5. Conditional G1 only after G0 PASS

```bash
HVALS=1,32,256,2048 bash run_g1_persistence.sh full 0,1,2,3,4
```

G1 cannot rescue a null.
