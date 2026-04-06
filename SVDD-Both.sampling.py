import os
import json
import glob
import subprocess
from datetime import datetime

LOGFILE = "svdd-both.sampling.runlog.txt"   # global log file

def log(msg):
    """Append a timestamped message to the global log file and print it."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOGFILE, "a") as f:
        f.write(line + "\n")

def pick_least_used_gpu(default_gpu=0):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=True
        )
        usages = []
        for line in result.stdout.strip().splitlines():
            idx_str, mem_str = [x.strip() for x in line.split(",")]
            usages.append((int(idx_str), int(mem_str)))
        usages.sort(key=lambda x: x[1])
        best_gpu = usages[0][0]
        log(f"[GPU picker] Selected GPU {best_gpu}")
        return best_gpu
    except Exception as e:
        log(f"[GPU picker] Error, using default GPU {default_gpu}: {e}")
        return default_gpu

# settings
sampling_mode = "svdd_pm"
samples = 30
sample_M = 10
select = "softmax"
temp = 0.5
windows = {
    "PM-Late": {"branch_start":0, "branch_end":120},
    #"PM-Mid": {"branch_start":120, "branch_end":300}
}
weights = {
    "SVDD-Geom": 1.0, #Geometry-biased: SA/Geom=0.0/1.0
    "SVDD-SA": 0.0,  # SA-biased: SA/Geom=1.0/0.0
    "SVDD-Both": 0.5 # equal weights: SA/Geom=0.5/0.5
}
exp_name = "SVDD-Both" # options: SVDD-SA, SVDD-Geom
w_rigid = weights[exp_name]


# Inputs
input_json = "/isilon/ytang4/protac_design/database/independent_PROTACpedia_linker.json"

# Outputs
save_dir = "/isilon/ytang4/protac_design/svdd-protac/PROTACpedia"
log_dir = "/isilon/ytang4/protac_design/svdd-protac/logs"

# Per-window sampling
for w, b in windows.items():
    output_dir = f"{save_dir}/{exp_name}/temp_{temp}/{w}"
    os.makedirs(output_dir, exist_ok=True)
    log(f"Output folder created: {output_dir}")
    branch_start, branch_end = b['branch_start'], b['branch_end']
    gpu_id = pick_least_used_gpu(default_gpu=0)
    cmd = [
            "/isilon/ytang4/tools/miniconda3/envs/protacs/bin/python", "src/make_inference_w_guidance.py",
            "--test", input_json,
            "--mode", sampling_mode,
            "--samples", str(samples),
            "--sample_M", str(sample_M),
            "--select", select,
            "--temperature", str(temp),
            "--branch_start", str(branch_start),
            "--branch_end", str(branch_end),
            "--gpu", str(gpu_id),
            "--save_dir", output_dir,
            "--w_rigid", str(w_rigid),
            "--no_wandb",
    ]

    # Per-structure log file
    pid = os.getpid()
    logfile = f"{log_dir}/{exp_name}.temp_{temp}.{w}.pid_{pid}.sampling.runlog"

    # Start timestamp
    start = datetime.now()
    log(f"START {exp_name}.temp_{temp}.{w} | GPU {gpu_id} | PID{pid}")
    log("CMD: " + " ".join(cmd))

    # Write header and stream stdout/stderr
    with open(logfile, "a") as lf:
        lf.write(f"Started at {start}\n")
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        subprocess.run(cmd, stdout=lf, stderr=lf, check=True)

    # End timestamp
    end = datetime.now()
    delta = end - start

    log(f"END   | Duration: {delta}")
    log("-" * 120)
