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
from replay_memory import ReplayMemory
from LogicSynthesisEnv import NUM_LENGTH_EPISODES, LogicSynthesisEnv
from mcts import execute_episode,test_episode
import argparse,os
import os.path as osp
import zipfile
from zipfile import ZipFile
import torch,shutil
import statistics
from LogicSynthesisPolicy import LogicSynthesisPolicy

def log(iteration, actionList, total_rew):
    time.sleep(0.3)
    print(f"Training Episodes: {iteration}")
    for p in actionList:
        print("Action prob:",p)
        print("Action chosen:",np.argmax(p))
    print(f"Return: {total_rew}")
    
def main():
    
    parser = argparse.ArgumentParser(description='MCTS+RL')
    parser.add_argument('--benchmark', type=str, required=True,
                        help='Path of design AIG')
    parser.add_argument('--library', type=str, required=True,
                        help='Technology library path')
    parser.add_argument('--dumpdir', type=str, required=True,default="",
                        help='Main DUMP directory of benchmark to store result')
    parser.add_argument('--runID', type=int, required=True, default=0,
                        help='Run ID of SA run')
    parser.add_argument('--budget', type=int, required=False, default=100,
                        help='Max. budget iterations')
    parser.add_argument('--model', type=str, required=False, default="",
                        help='Pre-trained graph')
    args = parser.parse_args()
    
    origAIGPath = args.benchmark
    libraryPath = args.library
    rootDumpDir = args.dumpdir
    runID = args.runID
    max_budget = args.budget
    preTrainedGraphModel = args.model
    
    if not (osp.exists(origAIGPath) and osp.exists(libraryPath)):
        print("Incorrect path. Rerun")
        exit(0)

    if not(osp.exists(preTrainedGraphModel)):
        preTrainedGraphModel = None
        print("Pre trained model path does not exist.")
        exit(0)
    
    runFolder = osp.join(rootDumpDir,'run'+str(runID))
    csvResultFile = osp.join(rootDumpDir,'data_run'+str(runID)+".csv") # Data to store area obtained in each iteration and best so far
    logFilePath = osp.join(runFolder,'log_run'+str(runID)+".log")   # Log data to dump area and read
    #finalAIGPath = osp.join(runFolder,'aig_runID'+str(runID)+".aig")
    if not osp.exists(rootDumpDir):
        os.mkdir(rootDumpDir)
        
    #if osp.exists(runFolder):
    #    shutil.rmtree(runFolder)
    if not osp.exists(runFolder):    
        os.mkdir(runFolder)
        
    benchmarkDesignName = os.path.splitext(os.path.basename(origAIGPath))[0]
    destRootAIGPath = osp.join(runFolder,benchmarkDesignName+"+0+step0.aig")
    shutil.copy(origAIGPath,destRootAIGPath)
    
    def getPtData(state,ptZipFile,test_env):
        if os.path.exists(ptZipFile):
            filePathName = osp.basename(osp.splitext(ptZipFile)[0])
            with ZipFile(ptZipFile) as myzip:
                with myzip.open(filePathName) as myfile:
                    data = torch.load(myfile)
        else:
            ptFilePath = ptZipFile.split('.zip')[0]
            data = test_env.extract_pytorch_graph(state)
            torch.save(data,ptFilePath)
            with zipfile.ZipFile(ptZipFile,'w',zipfile.ZIP_DEFLATED) as fzip:
                fzip.write(ptFilePath,arcname=osp.basename(ptFilePath))
            os.remove(ptFilePath)
        return [data]

    def test_agent(trainedNetwork,iteration):
        test_env = LogicSynthesisEnv(origAIG=destRootAIGPath,logFile=logFilePath,libFile=libraryPath)
        state, _, done, _ = test_env.reset()
        step_idx = 0
        actionList = []
        while not done:
            pt_state = os.path.splitext(state)[0]+'.pt.zip'
            inputState = getPtData(state,pt_state,test_env)
            #p, _ = network.step(inputState)
            p, _,aigEmbed = trainedNetwork.stepNgetEmbedding(inputState)
            action = np.argmax(p[0])
            print("State:",pt_state)
            print("AIG embed:",aigEmbed)
            print("Action probs:",p)
            print("Action taken:",action)
            state, _, done, _ = test_env.step(state,step_idx,action)
            #state = next_state
            step_idx+=1
        returnVal = test_env.get_factor_return(state)
        log(iteration, actionList, returnVal)
            
    def plotFunction(lossArr,lossMetric,idx):
        plt.clf()
        plt.plot(lossArr, label=lossMetric)
        dumpPlotFile = osp.join(runFolder,"plot_"+lossMetric+"_"+str(idx)+".png")
        plt.legend()
        plt.savefig(dumpPlotFile,bbox_inches='tight')
            
    def test_network():
        modelPath = preTrainedGraphModel #osp.join(runFolder,preTrainedGraphModel)
        if not osp.exists(modelPath):
            print("Pre-trained model not found")
            exit(1)
        #preTrainedNetwork = LogicSynthesisPolicy(readout_type=['sum','max'])
        #preTrainedNetwork = LogicSynthesisPolicy()
        #preTrainedNetwork.load_state_dict(torch.load(osp.join(runFolder,modelPath)))
        #preTrainedNetwork.to('cuda')
        trainer = Trainer(preTrainedGraphModel=modelPath)
        simulationBudget = max_budget
        #test_agent(preTrainedNetwork,0)
        test_episode(trainer,simulationBudget,csvResultFile,LogicSynthesisEnv(origAIG=destRootAIGPath,logFile=logFilePath,libFile=libraryPath))
    
    test_network()
    
if __name__ == '__main__':
    main()