import os
import sys
import time
import argparse
import datetime

def gpu_info(gpu_index=0):
    try:
        info = os.popen('nvidia-smi|grep %').read().split('\n')[gpu_index].split('|')
    except IOError:
        print("IO error")
        exit()

    power = int(info[1].split()[-3][:-1])
    memory = int(info[2].split('/')[0].strip()[:-3])

    return power, memory

def narrow_setup(args):
    gpu_power, gpu_memory = gpu_info(args.gpu_index)
    i = 0
    while gpu_memory > 100:  # set waiting condition
        print("waiting...")
        gpu_power, gpu_memory = gpu_info(args.gpu_index)
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_power_str = 'gpu power:%d W |' % gpu_power
        gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
        sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
        sys.stdout.flush()
        time.sleep(args.interval)
        i += 1


    print('\n' + args.cmd)
    os.system(args.cmd)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', default=2, type=int)
    parser.add_argument('--gpu_index', default=0, type=int)
    parser.add_argument('--cmd', default='cd /root/whzh/codes/HOI/upt/experiments/hico/pretrain && sh run.sh', 
                        type=str)
    args = parser.parse_args()
    start_time = time.time()
    narrow_setup(args)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('waiting time {}'.format(total_time_str))