import os

os.system(f"python fine_tune_bart.py tldr 0.00003 161 \
            --batch_size 16 \
            --grad_accumulation_steps 8")

os.system(f"python fine_tune_bart.py tosdr 0.00003 161 \
            --batch_size 16 \
            --grad_accumulation_steps 8")

os.system(f"python fine_tune_bart.py small_billsum 0.00003 161 \
            --batch_size 16 \
            --grad_accumulation_steps 8")

os.system(f"python fine_tune_bart.py tiny_billsum 0.00003 161 \
            --batch_size 16 \
            --grad_accumulation_steps 8")
