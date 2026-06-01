import os
import glob
import re

for folder in glob.glob('logs/rsl_rl/*/*'):
    if not os.path.isdir(folder):
        continue
    models = glob.glob(os.path.join(folder, 'model_*.pt'))
    if not models:
        continue
    
    # Extract numbers
    def get_num(f):
        m = re.search(r'model_(\d+)\.pt', f)
        return int(m.group(1)) if m else -1
        
    models.sort(key=get_num)
    
    best_model = models[-1]
    for m in models[:-1]:
        os.remove(m)
        print(f"Deleted {m}")
    print(f"Kept {best_model}")
