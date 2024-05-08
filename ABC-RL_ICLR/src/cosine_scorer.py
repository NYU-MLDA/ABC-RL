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
from mcts import execute_episode,test_episode,get_init_graphEmbedding
import argparse,os
import os.path as osp
import zipfile
from zipfile import ZipFile
import torch,shutil
import statistics,random
from LogicSynthesisPolicy import LogicSynthesisPolicy
from joblib import Parallel, delayed
from tqdm import tqdm
from torch.nn import functional as tf
from basic_FR_logits import *
import pandas as pd
import seaborn as sb
#from energy import *
#from softmax_response import *
from scipy.spatial.distance import euclidean,cosine


BENCHMARK_DIR="/home/abc586/currentResearch/TCAD_RL_Synthesizor/benchmarks/arithmetic"


trainTestBenchmarkSplit = {
  0 : ['alu2', 'apex3', 'apex5', 'b2', 'c1355', 'c5315', 'c2670', 'prom2', 'frg1', 'i7', 'i8', 'm3', 'max512', 'table5','adder','div','log2','sin','sqrt','multiplier','max','square','bar','cavlc','ctrl','i2c','int2float','mem_ctrl','priority','router','arbiter','voter'],
  1 : ['alu4','apex1','apex2','apex4','apex7','b9','c880','c1908','c3540','c6288','c7552','frg2','i9','i10','m4','max128','prom1','pair','max1024'],
  2 : ['adder','bar','div','log2','max','multiplier','sin','square','sqrt'],
  3 : ['arbiter','cavlc','ctrl','i2c','int2float','mem_ctrl','priority','router','voter'],
  4 : ['adder_alu2','bar_alu2','div_alu2','log2_alu2','max_alu2','multiplier_alu2','sin_alu2','square_alu2','sqrt_alu2'],
  5 : ['adder_apex7','bar_apex7','div_apex7','log2_apex7','max_apex7','multiplier_apex7','sin_apex7','square_apex7','sqrt_apex7'],
}

sequenceEmbeddings = []


def loadSequenceEmbeddings(fileName):
    global sequenceEmbeddings
    with open(fileName,'r') as f:
        lines = f.readlines()
        for l in lines:
            stepAndSequences = l.strip("\r\n")
            sequenceEmbeddings.append(stepAndSequences)
    
def log(iteration, actionList, total_rew):
    time.sleep(0.3)
    print(f"Training Episodes: {iteration}")
    for p in actionList:
        print("Action prob:",p)
        print("Action chosen:",np.argmax(p))
    print(f"Return: {total_rew}")

def seed_everything(seed=566):                                                 
    random.seed(seed)
    #torch.seed(seed)
    #torch.use_deterministic_algorithms(True)                                                 
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
    args = parser.parse_args()
    
    #trainTestSplit = args.ttsplit
    libraryPath = args.library
    rootDumpDir = args.dumpdir
    runID = args.runID
    max_iterations = args.runs
    preTrainedModel = args.model
    #replayMemSize = args.replay
    #batchsize=args.bs
    
    
    seed_everything()
    if not (osp.exists(libraryPath)):
        print("Incorrect path. Rerun")
        exit(0)
    
    if not(osp.exists(preTrainedModel)):
        preTrainedModel = None
        print("WARNING: Pre-trained graph model doesn't exist")
        exit(0)
    
    runFolder = osp.join(rootDumpDir,'run'+str(runID))
    csvResultFile = osp.join(rootDumpDir,'data_run'+str(runID)+".csv") # Data to store area obtained in each iteration and best so far
    logFilePath = osp.join(runFolder,'log_run'+str(runID)+".log")   # Log data to dump area and read
    if not osp.exists(rootDumpDir):
        os.mkdir(rootDumpDir)
        
    if not osp.exists(runFolder):    
        os.mkdir(runFolder)
        
    
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
        test_env = LogicSynthesisEnv(origAIG=destRootAIGPaths[0][idx],logFile=logFilePath,libFile=libraryPath)
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
        
    
        
    # def getEmbedding(aigPath):
    #     return get_init_graphEmbedding(trainer,LogicSynthesisEnv(origAIG=aigPath,logFile=logFilePath,libFile=libraryPath))
    
    
    def getPtEmbedding(ptZipFile,init_gd,seqSentence,test_env):
        if os.path.exists(ptZipFile):
            filePathName = osp.basename(osp.splitext(ptZipFile)[0])
            with ZipFile(ptZipFile) as myzip:
                with myzip.open(filePathName) as myfile:
                    data = torch.load(myfile)
        else:
            ptFilePath = ptZipFile.split('.zip')[0]
            data = test_env.createDataFromGraphAndSequence(seqSentence,graphData=init_gd)
            torch.save(data,ptFilePath)
            with zipfile.ZipFile(ptZipFile,'w',zipfile.ZIP_DEFLATED) as fzip:
                fzip.write(ptFilePath,arcname=osp.basename(ptFilePath))
            os.remove(ptFilePath)
        return [data]
    
    def getLogits(aigPath):
        outputLogits = []
        outputTargets = []
        lsEnv = LogicSynthesisEnv(origAIG=aigPath,logFile=logFilePath,libFile=libraryPath)
        init_graph_data = lsEnv.extract_init_graph(aigPath)
        for idx,seq in enumerate(sequenceEmbeddings):
            print(seq)
            ptZipFile = osp.splitext(aigPath)[0]+"_seq"+str(idx)+".pt.zip"
            data = getPtEmbedding(ptZipFile,init_graph_data,seq,lsEnv)
            pi,_,logits = trainer.stepNgetLogits(data)
            outputLogits.append(logits)
            outputTargets.append(np.argmax(pi))
        return outputLogits,outputTargets
    
    def getAigEmbedding(aigPath):
        lsEnv = LogicSynthesisEnv(origAIG=aigPath,logFile=logFilePath,libFile=libraryPath)
        state, _, done, _ = lsEnv.reset()
        init_graph_data = lsEnv.extract_init_graph(aigPath)
        pt_state = osp.splitext(aigPath)[0]+".pt.zip"
        inputState = getPtData(state,pt_state,init_graph_data,lsEnv)
        p, _,aigEmbed = trainer.stepNgetEmbedding(inputState)
        return aigEmbed   
        
    
    def get_perplexity_histplot(df=pd.DataFrame,fileName="histogramForOOD.png"):
        sb.diverging_palette(220, 20, as_cmap=True)
        fig = plt.figure(dpi=300)
        ax = sb.histplot(data=df,x='scores',hue='category',element="step",palette='Set2')
        plt.title('Perplexity histogram')
        plt.savefig(runFolder+fileName)
        plt.show()
        return ax
        
   
    value_losses = []
    policy_losses = []
    modelPath = osp.join(runFolder,preTrainedModel)
    if not osp.exists(modelPath):
        print("Pre-trained model not found")
        exit(1)
    #trainer = Trainer(preTrainedGraphModel=modelPath)
    trainer = Trainer(preTrainedGraphModel=None)
    
    destRootAIGPaths = {} #Split for train and test AIGs
    for idx in range(len(trainTestBenchmarkSplit.keys())):
        destRootAIGPaths[idx] = []
        for b in trainTestBenchmarkSplit[idx]:
            origAbsPath = osp.join(BENCHMARK_DIR,b+'.aig')
            destRootAIG = osp.join(runFolder,b+"+0+step0.aig")
            shutil.copy(origAbsPath,destRootAIG)
            destRootAIGPaths[idx].append(destRootAIG)
    

    refDesEmbeddingFile = open(osp.join(rootDumpDir,"refDesign.txt"),'w')
    with open(osp.join(rootDumpDir,"embeddingDistance.csv"),'w') as f:
        refDesEmbedding = []
        f.write("benchmark")
        for idx in range(len(trainTestBenchmarkSplit[0])):  # Original Designs
            refDesName = trainTestBenchmarkSplit[0][idx]
            refDesAigPath = destRootAIGPaths[0][idx]
            aigEmbedding = getAigEmbedding(refDesAigPath)
            refDesEmbedding.append(aigEmbedding)
            #refDesEmbeddingFile = open(osp.join(rootDumpDir,refDesName+".out"))
            np.savetxt(osp.join(rootDumpDir,refDesName+".out"),aigEmbedding)
            # refDesEmbeddingFile.write(refDesName)
            # refDesEmbeddingFile.write(str(aigEmbedding.tostring()))
            # refDesEmbeddingFile.write("\n\n")
            f.write(","+refDesName)
        f.write("\n")
        for idx in range(1,6):
            catList = trainTestBenchmarkSplit[idx]
            for idx2 in range(len(catList)):
                desName = catList[idx2]
                desUCAIGPath = destRootAIGPaths[idx][idx2]
                embUC = getAigEmbedding(desUCAIGPath)
                f.write(desName)
                for refEmb in refDesEmbedding:
                    euclidDistance = cosine(embUC,refEmb)
                    f.write(",{:.3f}".format(euclidDistance))
                f.write("\n")
    
