"""Run Parts A-D in order. Usage: python run_all.py   (after python generate_data.py)"""
from copilot import part_a, validate_a, part_b, part_c, part_d

for name, mod in (("A", part_a), ("A validation", validate_a), ("B", part_b), ("C", part_c), ("D", part_d)):
    print(f"\n===== PART {name} =====")
    mod.main()
