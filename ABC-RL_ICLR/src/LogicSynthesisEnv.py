import numpy as np
import gym
import os,re
import torch
from utils import cprint
from static_env import StaticEnv
import abc_py as abcPy


synthesisOpToPosDic = \
{
     0: "refactor",
     1: "refactor -z",
     2: "rewrite" ,
     3: "rewrite -z" ,
     4: "resub" ,
     5: "resub -z",
     6: "balance"
}

import torch_geometric
import torch_geometric.data
from transformers import BertTokenizer


NUM_LENGTH_EPISODES = 10
RESYN2_CMD =  "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;"
BERT_MODEL_NAME = 'bert-base-cased'
tz = BertTokenizer.from_pretrained(BERT_MODEL_NAME)


class LogicSynthesisEnv(gym.Env, StaticEnv):
    """
    Simple gym environment with the goal to navigate the player from its
    starting position to the highest point on a two-dimensional map within
    a limited number of steps. Rewards are defined as the difference in
    altitude between states minus a penalty for each step. The player starts
    in the lower left corner of the map and the highest point is in the upper
    right corner. Map layout mirrors CliffWalking environment:
    top left = (0, 0), top right = (0, m-1), bottom left = (n-1, 0),
    bottom right = (n-1, m-1).
    The setup of this environment was inspired by the energy landscape in
    protein folding.
    """

    n_actions = 7

    def __init__(self,origAIG,libFile,logFile):
        self._abc = abcPy.AbcInterface()
        self._abc.start()
        self.orig_aig = origAIG
        self.ep_length = NUM_LENGTH_EPISODES
        self.step_idx = 0
        self.lib = libFile
        self.logFile = logFile
        self.baselineReturn = self.getResynReturn()

    def initial_state(self):
        state = self.orig_aig
        return state

    def reset(self):
        state = self.orig_aig
        return state, 0, False, None

    def step(self, state, depth, action):
        assert action >= 0 and action < 7
        next_state = self.next_state(state,depth,action)
        done = (depth+1) == self.ep_length
        return next_state, 0, done, None

    
    def next_state(self,state,depth,action):
        fileDirName = os.path.dirname(state)
        fileBaseName,prefix,_ = os.path.splitext(os.path.basename(state))[0].split("+")
        nextState = os.path.join(fileDirName,fileBaseName+"+"+prefix+str(action)+"+step"+str(depth+1)+".aig")
        self._abc.read(state)
        if action == 0:
            self._abc.refactor(l=False, z=False) #rf
        elif action == 1:
            self._abc.refactor(l=False, z=True) #rf -z
        elif action == 2:
            self._abc.rewrite(l=False, z=False) #rw -z
        elif action == 3:
            self._abc.rewrite(l=False, z=True) #rw -z
        elif action == 4:
            self._abc.resub(k=8,n=1,l=False, z=False) #rs
        elif action == 5:
            self._abc.resub(k=8,n=1,l=False, z=True) #rs -z
        elif action == 6:
            self._abc.balance(l=False) #balance
        else:
            print(action)
            assert(False)
        self._abc.write(nextState)
        return nextState

    def is_done_state(self,step_idx):
        return step_idx == self.ep_length
    
    #@staticmethod
    #def get_obs_for_states(states):
    #    return np.array(states)
    
    def extract_init_graph(self,state):
        self._abc.read(state)
        data = {}
        numNodes = self._abc.numNodes()
        data['node_type'] = np.zeros(numNodes,dtype=int)
        data['num_inverted_predecessors'] = np.zeros(numNodes,dtype=int)
        edge_src_index = []
        edge_target_index = []
        for nodeIdx in range(numNodes):
            aigNode = self._abc.aigNode(nodeIdx)
            nodeType = aigNode.nodeType()
            data['num_inverted_predecessors'][nodeIdx] = 0
            if nodeType == 0 or nodeType == 2:
                data['node_type'][nodeIdx] = 0            
            elif nodeType == 1:
                data['node_type'][nodeIdx] = 1
            else:
                data['node_type'][nodeIdx] = 2
                if nodeType == 4:
                    data['num_inverted_predecessors'][nodeIdx] = 1
                if nodeType == 5:
                    data['num_inverted_predecessors'][nodeIdx] = 2
            if (aigNode.hasFanin0()):
                fanin = aigNode.fanin0()
                edge_src_index.append(nodeIdx)
                edge_target_index.append(fanin)
            if (aigNode.hasFanin1()):
                fanin = aigNode.fanin1()
                edge_src_index.append(nodeIdx)
                edge_target_index.append(fanin)
        data['edge_index'] = torch.tensor([edge_src_index,edge_target_index],dtype=torch.long)
        data['node_type'] = torch.tensor(data['node_type'])
        data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])
        #data = torch_geometric.data.Data.from_dict(data)
        #data.num_nodes = numNodes
        data['nodes'] = numNodes
        return data

    def extract_pytorch_graph(self,state,graphData=None):
        fileBaseName = os.path.splitext(os.path.basename(state))[0]
        firstName,seq,stepNum = fileBaseName.split('+')
        stepNum = stepNum.split('step')[-1]
        seqIndividual = [x for x in seq]
        if graphData == None:
            seqSentence = firstName+" "+stepNum +" "+" ".join(seqIndividual)
        else:
            seqSentence = stepNum +" "+" ".join(seqIndividual)
        data = self.createDataFromGraphAndSequence(seqSentence,graphData)
        return data
    
    def createDataFromGraphAndSequence(self,seqSentence,graphData=None):
        encoded = tz.encode_plus(
                text=seqSentence,  # the sentence to be encoded
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = 32,  # maximum length of a sentence
                pad_to_max_length=True,  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )
        data={}
        data['data_input'] = encoded['input_ids']
        data['data_attention'] = encoded['attention_mask']
        if not graphData == None:
            for k,v in graphData.items():
                data[k] = v
            numNodes = data['nodes']
            data = torch_geometric.data.Data.from_dict(data)
            data.num_nodes = numNodes
        else:
            data = torch_geometric.data.Data.from_dict(data)
        return data
        
    
    def getResynReturn(self):
        abcRunCmd = "yosys-abc -c \"read "+self.orig_aig+";"+RESYN2_CMD+"read_lib "+self.lib+"; map ; topo;stime \" > "+self.logFile
        os.system(abcRunCmd)
        with open(self.logFile) as f:
            areaInformation = re.findall('[a-zA-Z0-9.]+',f.readlines()[-1])
            return float(areaInformation[-9])*float(areaInformation[-4])
        
    def get_return(self,state):
        abcRunCmd = "yosys-abc -c \"read "+state+"; read_lib "+self.lib+"; map ; topo;stime \" > "+self.logFile
        os.system(abcRunCmd)
        with open(self.logFile) as f:
            areaInformation = re.findall('[a-zA-Z0-9.]+',f.readlines()[-1])
            #return self.baselineReturn/(float(areaInformation[-9])*float(areaInformation[-4]))
            adpVal = float(areaInformation[-9])*float(areaInformation[-4])
            return max(-1,((self.baselineReturn-adpVal)/self.baselineReturn))
    
    def get_factor_return(self,state):
        abcRunCmd = "yosys-abc -c \"read "+state+"; read_lib "+self.lib+"; map ; topo;stime \" > "+self.logFile
        os.system(abcRunCmd)
        with open(self.logFile) as f:
            areaInformation = re.findall('[a-zA-Z0-9.]+',f.readlines()[-1])
            return self.baselineReturn/(float(areaInformation[-9])*float(areaInformation[-4]))

    def get_montecarlo_return(self,startingAction,state,depth):
        synthesisCmd=synthesisOpToPosDic[startingAction]+";"
        while(depth+1 <= self.ep_length):
            i = np.random.randint(0, len(synthesisOpToPosDic.keys()))
            synthesisCmd += (synthesisOpToPosDic[i]+";")
            depth+=1
        abcRunCmd = "yosys-abc -c \"read "+state+"; read_lib "+self.lib+";"+synthesisCmd+"map ; topo;stime \" > "+self.logFile
        os.system(abcRunCmd)
        with open(self.logFile) as f:
            areaInformation = re.findall('[a-zA-Z0-9.]+',f.readlines()[-1])
            #return self.baselineReturn/(float(areaInformation[-9])*float(areaInformation[-4]))
            adpVal = float(areaInformation[-9])*float(areaInformation[-4])
            return max(-1,((self.baselineReturn-adpVal)/self.baselineReturn))

if __name__ == '__main__':
    library = os.path.join("testBench","45nm.lib")
    benchFile = os.path.join("testBench","des3_area.aig")
    env = LogicSynthesisEnv(aig=benchFile,lib=library)
    #env.render()
    print(env.step(0))
    #env.render()
    print(env.step(1))
    #env.render()
    print(env.step(2))
    #env.render()
    print(env.step(3))
    #env.render()
    print(env.step(4))
    print(env.step(5))
    print(env.step(6))
