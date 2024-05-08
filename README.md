# ABC-RL (WIP)

Pre-requisite:

ABC-RL has explicit dependency on abc_py (https://github.com/krzhu/abc_py). The repository contains the abc_rl_dependency folder used in this work to install abc_py and corresponding pybind11 plugins.

Steps:

1) ABC-RL_ICLR is the GCN implementation of the network architecture submitted in ICLR. ABC-RL_DAGNN is the advanced GNN architecture suited for circuits.

2) arithemetic folder contain the circuit AIG graphs.

3) In both ABC-RL_ICLR and ABC-RL_DAGNN, update the path of benchmarks folder in main.py (Line 24)

4) Modify the shell script file mcts_agent_training.sh with proper paths. Update the library with proper lib folder path attached with this repository.

5) Install yosys along with yosys-abc and set it in the path.

6) Run mcts_agent_training.sh

7) ABC-RL_ICLR will take around 10-12 days of time for training. ABC-RL_DAGNN will take around 5-6x time compared to ABC-RL_ICLR (estimated on A4000 GPU without parallelization. Under testing)

Make sure to have considerable space availability. For one training, 350GB-550GB of data will be generated. Thus, if parallel trainings are performed on same machine, ensure around 1TB of space is available.

Evaluation with pre-trained models:

1) Run evaluate_test_circuit.sh by appropriately modifying the ${ROOT} path. The improvement over resyn2 in terms of ADP (i.e. ADP of resyn2/ADP of ABC-RL recipe) will be dumped as data_runXX.csv file inside the dumpdir of each run. The first column represents current ADP achievied in MCTS run and second column represents best ADP so far.
