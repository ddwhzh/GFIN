ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9

echo "kill python finish"

ps -ef | grep nvidia-smi | grep -v grep | awk '{print $2}' | xargs kill -9
