"""
Example program that uses the single-player MCTS algorithm to train an agent
to master the HillClimbingEnvironment, in which the agent has to reach the
highest point on a map.
"""
#import imp
import time
import numpy as np
import matplotlib.pyplot as plt

from trainer import Trainer
from torch_geometric.loader import DataLoader
from replay_memory import ReplayMemory
from LogicSynthesisEnv import NUM_LENGTH_EPISODES, LogicSynthesisEnv
from mcts import execute_episode,test_episode
import argparse,os
import os.path as osp
import zipfile
from zipfile import ZipFile
import torch,shutil
import statistics,random
from LogicSynthesisPolicy import LogicSynthesisPolicy
from joblib import Parallel, delayed
BENCHMARK_DIR="/home/abc586/currentResearch/TCAD_RL_Synthesizor/benchmarks/arithmetic"

trainTestBenchmarkSplit = {
    15 : [['adder','div','log2','sin','sqrt','multiplier'],['max','square','bar']],
    25 : [['max','square','bar','div','sin','multiplier'],['adder','sqrt','log2']],
    35 : [['adder','div','log2','sqrt','max','square','bar'],['multiplier','sin']],
    45 : [['adder','log2','sqrt','square','bar','multiplier','sin'],['div','max']],
    1001 : [['bar','div','log2','multiplier','max','square','sin','sqrt'],['adder']],
    1002 : [['adder','div','log2','multiplier','max','square','sin','sqrt'],['bar']],
    1003 : [['adder','bar','log2','multiplier','max','square','sin','sqrt'],['div']],
    1004 : [['adder','bar','div','multiplier','max','square','sin','sqrt'],['log2']],
    1005 : [['adder','bar','div','log2','max','square','sin','sqrt'],['multiplier']],
    1006 : [['adder','bar','div','log2','multiplier','square','sin','sqrt'],['max']],
    1007 : [['adder','bar','div','log2','multiplier','max','square','sqrt'],['sin']],
    1008 : [['adder','bar','div','log2','multiplier','max','sin','sqrt'],['square']],
    1009 : [['adder','bar','div','log2','multiplier','max','square','sin'],['sqrt']],
    101 : [['adder'],['adder']],
    102 : [['bar'],['bar']],
    103 : [['div'],['div']],
    104 : [['log2'],['log2']],
    105 : [['multiplier'],['multiplier']],
    106 : [['max'],['max']],
    107 : [['sin'],['sin']],
    108 : [['square'],['square']],
    109 : [['sqrt'],['sqrt']],
    201 : [['cavlc','ctrl','dec','i2c','int2float','mem_ctrl','priority','router'],['arbiter','voter']],
    202 : [['arbiter','ctrl','dec','i2c','int2float','mem_ctrl','priority','voter'],['cavlc','router']],
    203 : [['arbiter','cavlc','dec','i2c','int2float','mem_ctrl','router','voter'],['ctrl','priority']],
    204 : [['arbiter','cavlc','ctrl','i2c','int2float','priority','router','voter'],['dec','mem_ctrl']],
    205 : [['arbiter','cavlc','ctrl','dec','mem_ctrl','priority','router','voter'],['i2c','int2float']],
    19 : [['div', 'multiplier', 'log2','sqrt','square'],['alu2']],
    20 : [['alu2', 'apex3', 'apex5', 'b2', 'c1355', 'c5315', 'c2670', 'prom2', 'frg1', 'i7', 'i8', 'm3', 'max512', 'table5'],['sin']],
    2001 : [['train10_r1','train12_r1','train13_r1','train14_r1'],['sqrt']],
    5001 : [['alu2', 'apex3', 'apex5', 'b2', 'c1355', 'c5315', 'c2670', 'prom2', 'frg1', 'i7', 'i8', 'm3', 'max512', 'table5','adder','log2','multiplier','max','ctrl','int2float','priority','arbiter'],['sin']]
}


def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def log(iteration, actionList, total_rew):
    time.sleep(0.3)
    print(f"Training Episodes: {iteration}")
    for p in actionList:
        print("Action prob:",p)
        print("Action chosen:",np.argmax(p))
    print(f"Return: {total_rew}")

def seed_everything(seed=566):                                                 
    random.seed(seed)                                   
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)                                                   
        torch.cuda.manual_seed_all(seed)                                             
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False                     

if __name__ == '__main__':
    n_actions = 7
    node_enc_outdim = 3
    gnn_hidden_dim = 32
    num_gcn_layer = 2
    
    
    parser = argparse.ArgumentParser(description='MCTS+RL')
    parser.add_argument('--ttsplit', type=int, required=True,
                        help='Train test split')
    parser.add_argument('--library', type=str, required=True,
                        help='Technology library path')
    parser.add_argument('--dumpdir', type=str, required=True,default="",
                        help='Main DUMP directory of benchmark to store result')
    parser.add_argument('--runID', type=int, required=True, default=0,
                        help='Run ID of SA run')
    parser.add_argument('--runs', type=int, required=False, default=1000,
                        help='Max. iterations')
    parser.add_argument('--model', type=str, required=False, default="",
                        help='Pre-trained model filename')
    parser.add_argument('--replay', type=int, required=False, default=10,
                        help='Replay memory size')
    parser.add_argument('--bs', type=int, required=False, default=32,
                        help='Batchsize')
    parser.add_argument('--critic', default=False, action=argparse.BooleanOptionalAction, help="Critic activation")
    #parser.add_argument('--critic', type=int, default=0,required=False,
    #                    help='Critic activation')


    args = parser.parse_args()
    
    trainTestSplit = args.ttsplit
    libraryPath = args.library
    rootDumpDir = args.dumpdir
    runID = args.runID
    max_iterations = args.runs
    preTrainedModel = args.model
    replayMemSize = args.replay
    batchsize=args.bs
    #isCritic = False if args.critic==0 else True
    isCritic = args.critic
    
    seed_everything()
    if not (osp.exists(libraryPath)):
        print("Incorrect path. Rerun")
        exit(0)

    
    if not(osp.exists(preTrainedModel)):
        preTrainedModel = None
        print("WARNING: Pre-trained graph model doesn't exist")
    
    runFolder = osp.join(rootDumpDir,'run'+str(runID))
    csvResultFile = osp.join(rootDumpDir,'data_run'+str(runID)+".csv") # Data to store area obtained in each iteration and best so far
    logFilePath = osp.join(runFolder,'log_run'+str(runID)+".log")   # Log data to dump area and read
    if not osp.exists(rootDumpDir):
        os.mkdir(rootDumpDir)
        
    if not osp.exists(runFolder):    
        os.mkdir(runFolder)
        
    destRootAIGPaths = [[],[]] #Split for train and test AIGs
    for idx in range(2):
        for b in trainTestBenchmarkSplit[trainTestSplit][idx]:
            origAbsPath = osp.join(BENCHMARK_DIR,b+'.aig')
            destRootAIG = osp.join(runFolder,b+"+0+step0.aig")
            shutil.copy(origAbsPath,destRootAIG)
            destRootAIGPaths[idx].append(destRootAIG)
    
    trainer = Trainer(preTrainedGraphModel=preTrainedModel,batch_size=batchsize,isCritic=isCritic)
    network = trainer.step_model
    if isCritic:
        print("\nCritic loss added\n")

    mem = ReplayMemory(replayMemSize,
                       { "ob": "string",
                         "pi": np.float32,
                         "return": np.float32},
                       { "ob": [],
                         "pi": [n_actions],
                         "return": []},batch_size=batchsize)

    
    
    def getPtData(state,ptZipFile,init_gd,test_env):
        if os.path.exists(ptZipFile):
            filePathName = osp.basename(osp.splitext(ptZipFile)[0])
            with ZipFile(ptZipFile) as myzip:
                with myzip.open(filePathName) as myfile:
                    data = torch.load(myfile)
        else:
            ptFilePath = ptZipFile.split('.zip')[0]
            data = test_env.extract_pytorch_graph(state,init_gd)
            torch.save(data,ptFilePath)
            with zipfile.ZipFile(ptZipFile,'w',zipfile.ZIP_DEFLATED) as fzip:
                fzip.write(ptFilePath,arcname=osp.basename(ptFilePath))
            os.remove(ptFilePath)
        return [data]

    def test_agent(iteration,idx):
        bLogFilePath = osp.join(rootDumpDir,'log_'+trainTestBenchmarkSplit[trainTestSplit][0][idx]+'run'+str(runID)+".log")
        test_env = LogicSynthesisEnv(origAIG=destRootAIGPaths[0][idx],logFile=bLogFilePath,libFile=libraryPath)
        state, _, done, _ = test_env.reset()
        step_idx = 0
        actionList = []
        init_graph_data = test_env.extract_init_graph(state)
        while not done:
            pt_state = os.path.splitext(state)[0]+'.pt.zip'
            inputState = getPtData(state,pt_state,init_graph_data,test_env)
            p, _,aigEmbed = trainer.stepNgetEmbedding(inputState)
            action = np.argmax(p[0])
            print("State:",pt_state)
            print("AIG embed:",aigEmbed[0][:20])
            print("Action probs:",p[0])
            print("Action taken:",action)
            state, _, done, _ = test_env.step(state,step_idx,action)
            step_idx+=1
        returnVal = test_env.get_factor_return(state)
        log(iteration, actionList, returnVal)
        
    def plotFunction(lossArr,lossMetric,idx):
        plt.clf()
        plt.plot(lossArr, label=lossMetric)
        dumpPlotFile = osp.join(runFolder,"plot_"+lossMetric+"_"+str(idx)+".png")
        plt.legend()
        plt.savefig(dumpPlotFile,bbox_inches='tight')
    
    def test_agent_after_episodes(trainedNetwork,iteration,idx):
        bLogFilePath = osp.join(rootDumpDir,'log_'+trainTestBenchmarkSplit[trainTestSplit][0][idx]+'run'+str(runID)+".log")
        csvFileAfterEpisodes = osp.join(rootDumpDir,'data_'+trainTestBenchmarkSplit[trainTestSplit][0][idx]+'_after_episode'+str(iteration)+".csv")
        simulationBudget = 100
        test_episode(trainedNetwork,simulationBudget,csvFileAfterEpisodes,LogicSynthesisEnv(origAIG=destRootAIGPaths[0][idx],logFile=bLogFilePath,libFile=libraryPath))
        
    # def test_network():
    #     modelPath = osp.join(runFolder,preTrainedModel)
    #     if not osp.exists(modelPath):
    #         print("Pre-trained model not found")
    #         exit(1)
    #     preTrainedNetwork = LogicSynthesisPolicy(readout_type=['sum','max'])
    #     preTrainedNetwork.load_state_dict(torch.load(osp.join(runFolder,modelPath)))
    #     simulationBudget = 90
    #     test_episode(preTrainedNetwork,simulationBudget,csvResultFile,LogicSynthesisEnv(origAIG=destRootAIGPaths[1][0],logFile=logFilePath,libFile=libraryPath))
    
    def collect_experiences(trainedNetwork,idx):
        bLogFilePath = osp.join(rootDumpDir,'log_'+trainTestBenchmarkSplit[trainTestSplit][0][idx]+'run'+str(runID)+".log")
        obs, pis, returns, done_state = execute_episode(trainedNetwork,LogicSynthesisEnv(origAIG=destRootAIGPaths[0][idx],logFile=bLogFilePath,libFile=libraryPath))
        return obs, pis, returns, done_state
    
    
    def parallelize_testing(trainedNetwork,iVal,idx):
        test_agent(iVal,idx)
        test_agent_after_episodes(trainedNetwork,iVal,idx)
        
        
   
    value_losses = []
    policy_losses = []


    for i in range(max_iterations):
        #network.eval() # Network always in evaluation mode, except train in training mode
        
        if i % 2 == 0 and i>0:
            # for idx_train_sample in range(len(destRootAIGPaths[0])):
            #     test_agent(i,idx_train_sample)
            #     test_agent_after_episodes(trainer,i,idx_train_sample)
            start = time.time()
            Parallel(n_jobs=5,verbose=10)(delayed(parallelize_testing)(trainer,i,idx_train_sample) for idx_train_sample in range(len(destRootAIGPaths[0])))
            end = time.time()
            simple_lapsed_time("\nTime taken for Evaluation", end-start)
            plotFunction(value_losses,"value_loss",i)
            plotFunction(policy_losses,'policy_loss',i)
            if i>0:
                torch.save(network.state_dict(),osp.join(runFolder,"nn_model_iter_{}.pt".format(i)))

        start = time.time()
        # for idx in range(len(destRootAIGPaths[0])):        
        #     obs, pis, returns, done_state = execute_episode(trainer,LogicSynthesisEnv(origAIG=destRootAIGPaths[0][idx],logFile=logFilePath,libFile=libraryPath))
        #     print(obs)
        #     mem.add_all({"ob": obs, "pi": pis, "return": returns})

        r = Parallel(n_jobs=5,verbose=10)(delayed(collect_experiences)(trainer,idx) for idx in range(len(destRootAIGPaths[0])))
        obsAll,pisAll,returnsAll,_ = zip(*r)
        
        for loopVar in range(len(destRootAIGPaths[0])):
            mem.add_all({"ob": obsAll[loopVar], "pi": pisAll[loopVar], "return": returnsAll[loopVar]})
        end = time.time()
        simple_lapsed_time("\nTime taken for MCTS", end-start)

                
        polLoss = []
        valLoss = []
        
        # Why 3? Batch size: 4 and 3 will ensure loosly all 10 states inserted in replay memory are collected.
        start = time.time()
        for idx in range(60):
            batch = mem.get_minibatch(order=0)
            vl, pl = trainer.train(batch["ob"], batch["pi"], batch["return"])
            polLoss.append(pl[0])
            valLoss.append(vl[0])
        
        avgPolLoss = np.mean(polLoss)
        avgValLoss = np.mean(valLoss)
        print("Value loss:"+str(avgValLoss))
        print("Policy loss:"+str(avgPolLoss))    
        value_losses.append(avgValLoss)
        policy_losses.append(avgPolLoss)
        trainer.scheduler_step(avgPolLoss)
        end = time.time()
        simple_lapsed_time("\nTime taken for training", end-start)
