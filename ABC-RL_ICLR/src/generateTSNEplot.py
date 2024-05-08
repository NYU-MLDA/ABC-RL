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
import seaborn as sns
from energy import *
from softmax_response import *
from scipy.spatial.distance import euclidean,cosine
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


BENCHMARK_DIR="/home/abc586/currentResearch/TCAD_RL_Synthesizor/benchmarks/arithmetic"
trainTestBenchmarkSplit = {
    3 :  ['adder','bar','div','log2','max','multiplier','sin','square','sqrt'],
    4 :  ['adder_alu2','bar_alu2','div_alu2','log2_alu2','max_alu2','multiplier_alu2','sin_alu2','square_alu2','sqrt_alu2'],
    5 :  ['adder_apex7','bar_apex7','div_apex7','log2_apex7','max_apex7','multiplier_apex7','sin_apex7','square_apex7','sqrt_apex7'],
    8 :  ['arbiter_alu2','cavlc_alu2','ctrl_alu2','i2c_alu2','int2float_alu2','mem_ctrl_alu2','priority_alu2','router_alu2','voter_alu2'],
    7 :  ['arbiter_apex7','cavlc_apex7','ctrl_apex7','i2c_apex7','int2float_apex7','mem_ctrl_apex7','priority_apex7','router_apex7','voter_apex7'],
    6 :  ['arbiter','cavlc','ctrl','i2c','int2float','mem_ctrl','priority','router','voter'],
    0 : ['alu2','apex3','apex5','b2','c1355','c5315','c2670','prom2','frg1','i7','i8','m3','max512','table5'],
    1 : ['alu4','apex1','apex2','apex4','b9','c880','c7552','frg2','i9','i10','m4','max128','prom1','max1024','pair'],
    2 : ['apex6','c432','c499','seq','table3','apex7','c1908','c3540','c6288'],
    #3 :  ['csela_orig','cra_orig','cskipa_orig','ksa_orig','max128','max_noopt_orig','mult_wallace_orig','mult_dadda_orig','div_noopt_orig','div_nooptlong_orig'],
}
sequenceEmbeddings = []


def plot_scatter(x, colors, fileName):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    print(num_classes)
    palette = np.array(sns.color_palette("hls", num_classes))
    # print(palette)
    # create a scatter plot.
    f = plt.figure(figsize=(16, 12))
    # ax = plt.subplot(aspect='equal')
    ax = plt.subplot()
    # sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=df['label'], cmap=plt.cm.get_cmap('Paired'))
    # sc = ax.scatter(x[:,0], x[:,1],  c=palette[colors.astype(np.int)], cmap=plt.cm.get_cmap('Paired'))
    sc = ax.scatter(x[:, 0], x[:, 1], c=palette[colors.astype(np.int)], cmap=plt.cm.get_cmap('Paired'))
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.legend()
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=5)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    ax.grid(True)
    plt.savefig(fileName + '.png',bbox_inches='tight')
    plt.show()
    
def gettSNEResults(features,labels):
    X = pd.DataFrame(features)
    Y = pd.DataFrame(labels)
    #X = X.sample(frac=0.1, random_state=10).reset_index(drop=True)
    #Y = Y.sample(frac=0.1, random_state=10).reset_index(drop=True)
    df = X
    time_start = time.time()
    tsne = TSNE(random_state=0)
    tsne_results = tsne.fit_transform(df.values)
    df['label'] = Y
    return tsne_results,df['label']


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
    parser.add_argument('--ttsplit', type=int, required=False, default=1020,
                        help='TTSPLIT identifier')
    args = parser.parse_args()
    
    #trainTestSplit = args.ttsplit
    libraryPath = args.library
    rootDumpDir = args.dumpdir
    runID = args.runID
    max_iterations = args.runs
    preTrainedModel = args.model
    ttID = args.ttsplit
    
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
        
   
    value_losses = []
    policy_losses = []
    modelPath = osp.join(runFolder,preTrainedModel)
    if not osp.exists(modelPath):
        print("Pre-trained model not found")
        exit(1)
    trainer = Trainer(preTrainedGraphModel=modelPath)
    
    destRootAIGPaths = {} #Split for train and test AIGs
    for idx in range(len(trainTestBenchmarkSplit.keys())):
        destRootAIGPaths[idx] = []
        for b in trainTestBenchmarkSplit[idx]:
            origAbsPath = osp.join(BENCHMARK_DIR,b+'.aig')
            destRootAIG = osp.join(runFolder,b+"+0+step0.aig")
            shutil.copy(origAbsPath,destRootAIG)
            destRootAIGPaths[idx].append(destRootAIG)
    

    # with open(osp.join(rootDumpDir,"embeddingDistance.csv"),'w') as f:
    #     refDesEmbedding = []
    #     f.write("benchmark")
    #     for idx in range(len(trainTestBenchmarkSplit[0])):  # Original Designs
    #         refDesName = trainTestBenchmarkSplit[0][idx]
    #         refDesAigPath = destRootAIGPaths[0][idx]
    #         aigEmbedding = getAigEmbedding(refDesAigPath)
    #         refDesEmbedding.append(aigEmbedding)
    #         f.write(","+refDesName)
    #     f.write("\n")
    #     for idx in range(1,4):
    #         catList = trainTestBenchmarkSplit[idx]
    #         for idx2 in range(len(catList)):
    #             desName = catList[idx2]
    #             desUCAIGPath = destRootAIGPaths[idx][idx2]
    #             embUC = getAigEmbedding(desUCAIGPath)
    #             f.write(desName)
    #             for refEmb in refDesEmbedding:
    #                 euclidDistance = cosine(embUC,refEmb)
    #                 f.write(",{:.3f}".format(euclidDistance))
    #             f.write("\n")
    
    embeddingList = []
    labelList = []
    ttSplitID = str(ttID)
    with open(osp.join(rootDumpDir,"designToLabel_ttsplit"+ttSplitID+".txt"),'w') as f:
        print(max_iterations)
        print(ttID)
        if (ttID==15) or (ttID==25) or (ttID==35) or (ttID==45):
            i=3
            j=6
        elif(ttID >=201) and (ttID <= 205):
            i=6
            j=9
        elif(ttID == 1020):
            i=0
            j=3
        else:
            print("Error. exiting")
            exit(1)
        counter = 0
        for idx in range(i,j):
            catList = trainTestBenchmarkSplit[idx]
            for idx2 in range(len(catList)):
                desName = catList[idx2]
                desUCAIGPath = destRootAIGPaths[idx][idx2]
                embUC = np.squeeze(getAigEmbedding(desUCAIGPath))
                f.write(desName+":"+str(counter)+"\n")
                embeddingList.append(embUC)
                labelList.append(counter)
                if not (idx == 0 or idx == 3 or idx == 6):
                    counter+=1
                
    tsne_F,tsne_Y = gettSNEResults(embeddingList,labelList)
    plot_scatter(tsne_F,tsne_Y,osp.join(rootDumpDir,"tSNE_designToLabel_ttsplit"+str(ttID)))
    