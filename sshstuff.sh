/HW-2/project cc@129.114.108.135:~/

ssh -i ~/.ssh/hmr-test.pem  cc@129.114.108.135

scp -i ~/.ssh/hmr-test.pem /Users/henriquerio/Documents/IIT/CS595-Project-FedRL/moonlander/experiment.py cc@129.114.108.135:~/moonlander/experiment.py
#copy full moonlander folder to ec2
scp -r -i ~/.ssh/hmr-test.pem /Users/henriquerio/Documents/IIT/CS595-Project-FedRL/moonlanderv3 cc@129.114.108.135:~/

#create screen session
screen -S test
#copy npz to sv_results/v1
scp -i ~/.ssh/hmr-test.pem cc@129.114.108.135:~/moonlander/dp_training_data_ep30_sens15.npz  /Users/henriquerio/Documents/IIT/CS595-Project-FedRL/moonlander/sv_results/v1
scp -i /Users/henriquerio/Documents/IIT/FALL2025-HW/CS595/hmr-test.pem cc@129.114.108.135:~/moonlander/federated_training_data.npz  /Users/henriquerio/Documents/IIT/CS595-Project-FedRL/moonlander/sv_results/v1
#continue screen session
screen -r test

#Detach from the screen session: Press Ctrl + a + d.

nohup python3 -u experiment.py > output_log.txt 2>&1 &
#process =37899
# check logs
tail -f output_log.txt

# kill all processes
pkill -u cc -9 python3