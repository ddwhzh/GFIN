#!/bin/bash
filename="exp"

chmod +x "./exps/"$filename".sh"
nohup "./exps/"$filename".sh" >> "./logs/train_"$filename".txt" 2>&1 & \
echo $! > "./logs/pid/save_pid.txt"