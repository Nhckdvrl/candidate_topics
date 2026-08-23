from __future__ import annotations
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g0_position_dissociation import build_target_program, normalize_line, parse_list

def test_generated_program_executes():
    p=build_target_program(random.Random(7),3,8); ns={}; exec(p.code,ns)
    assert ns['target_3'](p.input_x)==p.output
    assert len(p.trace)==8

def test_parse_list():
    assert parse_list('answer: [1, -2, 3]')==[1,-2,3]

def test_normalize_line():
    assert normalize_line('```python\n arr[2]   =   arr[1] + 4\n```')=='arr[2] = arr[1] + 4'
