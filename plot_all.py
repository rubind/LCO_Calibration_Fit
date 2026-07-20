import subprocess
import glob
import tqdm
import numpy as np

for dr in tqdm.tqdm(np.sort(glob.glob("runs/*/*"))):
    print(subprocess.getoutput("PYTHONPATH=src /opt/anaconda3/bin/python -m passband_fit plot --run " + dr))

