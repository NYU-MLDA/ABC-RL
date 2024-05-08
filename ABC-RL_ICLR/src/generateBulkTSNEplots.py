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
import pickle


BENCHMARK_DIR="/home/abc586/currentResearch/TCAD_RL_Synthesizor/benchmarks/arithmetic"
trainTestBenchmarkSplit = {
    3 :  ['adder','bar','div','log2','max','multiplier','sin','square','sqrt'],
    4 :  ['adder_alu2','bar_alu2','div_alu2','log2_alu2','max_alu2','multiplier_alu2','sin_alu2','square_alu2','sqrt_alu2'],
    5 :  ['adder_apex7','bar_apex7','div_apex7','log2_apex7','max_apex7','multiplier_apex7','sin_apex7','square_apex7','sqrt_apex7'],
    8 :  ['arbiter_alu2','cavlc_alu2','ctrl_alu2','i2c_alu2','int2float_alu2','mem_ctrl_alu2','priority_alu2','router_alu2','voter_alu2'],
    7 :  ['arbiter_apex7','cavlc_apex7','ctrl_apex7','i2c_apex7','int2float_apex7','mem_ctrl_apex7','priority_apex7','router_apex7','voter_apex7'],
    6 :  ['arbiter','cavlc','ctrl','i2c','int2float','mem_ctrl','priority','router','voter'],
    0 : ['alu2','apex3','apex5','b2','c1355','c5315','c2670','prom2','frg1','i7','i8','m3','max512','table5'],
    #1 : ['alu4','apex1','apex2','apex4','b9','c880','c7552','frg2','i9','i10','m4','max128','prom1','max1024','pair'],
    1 : ['c1908'],
    2 : ['apex6','c432','c499','seq','table3','apex7','c1908','c3540','c6288'],
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
        txt = ax.text(xtext, ytext, str(i), fontsize=15)
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
    parser.add_argument('--dumpdir', type=str, required=True,default="",
                        help='Main DUMP directory of benchmark to store result')
    args = parser.parse_args()
    
    rootDumpDir = args.dumpdir
    ttID = 1020
    

     
    seed_everything()
        
    #runFolder = osp.join(rootDumpDir,'run'+str(runID))
    #logFilePath = osp.join(runFolder,'log_run'+str(runID)+".log")   # Log data to dump area and read
    if not osp.exists(rootDumpDir):
        os.mkdir(rootDumpDir)
        
    #mcncDesigns = trainTestBenchmarkSplit[0]+trainTestBenchmarkSplit[1]+trainTestBenchmarkSplit[2]
    mcncDesigns = trainTestBenchmarkSplit[0]+trainTestBenchmarkSplit[1]
    tsneEmbeddingList = []
    tsneEmbeddingLabel = []
    tsneEmbeddingNameLabelDicts={}
    counter=0
    
    for k,d in enumerate(trainTestBenchmarkSplit[0]):
        if d == 'i8' or d == 'c6288':
            continue
        tsneEmbeddingNameLabelDicts[counter] = d
        for i in range(1,5):
            pklFile = osp.join(rootDumpDir,d+"_TTSPLIT_"+str(ttID)+"_"+str(i),d+"_"+str(i)+".pkl")
            if not osp.exists(pklFile):
                print(pklFile)
                continue
            with open(pklFile,'rb') as f:
                nameEmbeddingList = pickle.load(f)
            embeddingLists = nameEmbeddingList[1]
            #print(np.shape(embeddingLists[0]))
            tsneEmbeddingList.extend(embeddingLists)
            labelList = [counter for j in range(len(embeddingLists))]
            tsneEmbeddingLabel.extend(labelList)
        #counter+=1
        print(counter)
        print(d)
    
    counter+=1
    
    for k,d in enumerate(trainTestBenchmarkSplit[1]):
        if d == 'i8' or d == 'c6288':
            continue
        tsneEmbeddingNameLabelDicts[counter] = d
        for i in range(1,5):
            pklFile = osp.join(rootDumpDir,d+"_TTSPLIT_"+str(ttID)+"_"+str(i),d+"_"+str(i)+".pkl")
            if not osp.exists(pklFile):
                print(pklFile)
                continue
            with open(pklFile,'rb') as f:
                nameEmbeddingList = pickle.load(f)
            embeddingLists = nameEmbeddingList[1]
            #print(np.shape(embeddingLists[0]))
            tsneEmbeddingList.extend(embeddingLists)
            labelList = [counter for j in range(len(embeddingLists))]
            tsneEmbeddingLabel.extend(labelList)
        counter+=1
        print(counter)
        print(d)
            
    with open(osp.join(rootDumpDir,"designNameLabel.dict"),'w') as f:
        for k,v in tsneEmbeddingNameLabelDicts.items():
            f.write(str(k)+v)
            f.write("\n")
                
    tsne_F,tsne_Y = gettSNEResults(tsneEmbeddingList,tsneEmbeddingLabel)
    print(tsne_Y)
    plot_scatter(tsne_F,tsne_Y,osp.join(rootDumpDir,"tSNE_designToLabel_ttsplit"+str(ttID)))
    