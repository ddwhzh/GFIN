#!/bin/bash
filename="exp"

chmod +x "./exps/"$filename".sh"
nohup "./exps/"$filename".sh" >> "./logs/train_"$filename".txt" 2>&1 & echo $! > "./logs/pid/save_pid.txt"
#0 表示stdin标准输入
#1 表示stdout标准输出
#2 表示stderr标准错误
#2>&1 也就表示将错误重定向到标准输出上