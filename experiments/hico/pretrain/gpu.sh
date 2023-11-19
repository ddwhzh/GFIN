chmod +x "check_gpus.py"
nohup python check_gpus.py >> "./logs/out.txt" 2>&1 & echo $! > "./logs/pid/gpu_wait_pid.txt"