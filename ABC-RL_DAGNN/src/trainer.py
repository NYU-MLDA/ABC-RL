import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import sys,os
import os.path as osp
import numpy as np
from zipfile import ZipFile
from LogicSynthesisPolicy import LogicSynthesisPolicy


class Trainer:
    """
    Trainer for an MCTS policy network. Trains the network to minimize
    the difference between the value estimate and the actual returns and
    the difference between the policy estimate and the refined policy estimates
    derived via the tree search.
    """

    def __init__(self,preTrainedGraphModel=None,device='cuda',learning_rate=0.01,batch_size=32,isCritic=False):
        self.step_model = LogicSynthesisPolicy(readout_type=['mean','max']) # NYU-MLDA parameterize this
        self.device = device
        self.preTrainedGraphModel = preTrainedGraphModel
        self.batch_size = batch_size
        self.isCritic = isCritic
        
        
        # if not self.preTrainedGraphModel is None:
        #     ## Load all the keys from GCN layers trained for node prediction
        #     self.step_model.load_state_dict(torch.load(self.preTrainedGraphModel),strict=False)
        #     weightsForGradOn = ["denseLayer.weight", "denseLayer.bias", "dense_p1.weight", "dense_p1.bias", \
        #                         "dense_p2.weight", "dense_p2.bias", "dense_v1.weight", "dense_v1.bias", "dense_v2.weight", "dense_v2.bias"]
            
        #     ## Freezing all other layers except last few layers
        #     for pname,param in self.step_model.named_parameters():
        #         if pname in weightsForGradOn:
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = False
            
        #     self.step_model = self.step_model.to(self.device)
        #     #self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.step_model.parameters()),
        #     #                        lr=learning_rate,weight_decay=1e-5)
        #     self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.step_model.parameters()),
        #                             lr=learning_rate)
        #     # print("\n\nDEBUG information:")
        #     # for pname,p in self.step_model.named_parameters():
        #     #     if p.requires_grad:
        #     #         print(pname)
        #     #     else:
        #     #         print("Freeze: "+pname)
        #     #exit(0)            
        # else:
        #     #print("\n\nDEBUG information is false")
        #     self.step_model = self.step_model.to(self.device)
        #     #self.optimizer = torch.optim.Adam(self.step_model.parameters(),lr=learning_rate,weight_decay=1e-5)
        #     self.optimizer = torch.optim.Adam(self.step_model.parameters(),lr=learning_rate)
        #     print("CODE COMES HERE. ALL PARAMS SHOULD BE ON")
    
        self.step_model = self.step_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.step_model.parameters(),lr=learning_rate)
        self.value_criterion = nn.MSELoss()
        
        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        #self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=40)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',patience=4,verbose=True)
        
    def scheduler_step(self,val_loss):
        self.lr_scheduler.step(val_loss)

    def train(self,obs, search_pis, returns):
        search_pis = torch.from_numpy(search_pis)
        returns = torch.from_numpy(returns)
        graphDataList = []
        #print(obs)
        #print(len(obs))
        for ob in obs:
            if os.path.exists(ob):
                filePathName = osp.basename(osp.splitext(ob)[0])
                with ZipFile(ob) as myzip:
                    with myzip.open(filePathName) as myfile:
                        data = torch.load(myfile)
                graphDataList.append(data)
            else:
                print("Path")
                print(ob)
                print("Serious error check")
                exit(1)          
        valLoss = []
        policyLoss = []
        self.step_model.train()
        loader = DataLoader(graphDataList, batch_size=self.batch_size)
        count_dict = {}
        # for name, param in self.step_model.named_parameters():
        #     if 'weight' in name:
        #         if param.grad is None:
        #             count_dict[name] = None
        #             print(name,count_dict[name])
        #         else:
        #             count_dict[name] = torch.zeros(param.grad.shape)
        #             print(name,count_dict[name])

        loader = DataLoader(graphDataList, batch_size=self.batch_size)
        for _, batch in enumerate(tqdm(loader, desc="Iteration",file=sys.stdout)):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            logits, policy, value,_,_ = self.step_model(batch)
            logsoftmax = nn.LogSoftmax(dim=1)
            policy_loss = torch.mean(torch.sum(-search_pis.to(self.device)
                                                * logsoftmax(logits), dim=1))
            value_loss = self.value_criterion(value, returns.to(self.device))
            if self.isCritic:
                loss = policy_loss#+ value_loss #+ (0.01 * value_loss) Omitting value loss
            else:
                loss = policy_loss+value_loss #+ (0.01 * value_loss) Omitting value loss
            loss.backward()
            # for name, param in self.step_model.named_parameters():
            #     if 'weight' in name:
            #         if not param.grad is None:
            #             print(name,count_dict[name])
            #             temp = torch.zeros(param.grad.shape)
            #             temp[param.grad != 0] += 1
            #             print(temp)
            #             #count_dict[name] += temp
            self.optimizer.step()
            valLoss.append(value_loss.detach().cpu().numpy())
            policyLoss.append(policy_loss.detach().cpu().numpy())
            #print(policy_loss.detach().cpu().numpy())
            print(search_pis)
            print(logits.detach().cpu().numpy())
            print(batch.data_input)
            #grads = {n:p.requires_grad for n, p in self.step_model.named_parameters()}
            #print(grads)
            #print(self.step_model.denseLayer.weight)
           
        for name, param in self.step_model.named_parameters():
            if 'weight' in name:
                if not param.grad is None:
                    #print(name)
                    #temp = torch.zeros(param.grad.shape)
                    #temp[param.grad != 0] += 1
                    #temp[param.requires_grad == False] += 1
                    #temp2 = torch.zeros(param.grad.shape)
                    print(param.grad)
                    #print(temp)
                    #print(name,torch.sum(temp))
                    print(name,torch.sum(param.grad))
                    print("----------------------")
                    #count_dict[name] += temp
        
        #print("\n\nprinting weight update dicts")
        #print(count_dict)
        with torch.no_grad():
            torch.cuda.empty_cache()

        return valLoss,policyLoss
    
    def step(self,aigData,device='cuda'):
        """
        Returns policy and value estimates for given observations.
        :param obs: Array of shape [N] containing N observations.
        :return: Policy estimate [N, n_actions] and value estimate [N] for
        the given observations.
        """
        self.step_model.eval()
        loader = DataLoader(aigData, batch_size=self.batch_size)
        batchedAIG = next(iter(loader))
        batchedAIG = batchedAIG.to(device)
        with torch.no_grad():
            _, pi, v,_,_ = self.step_model(batchedAIG)
            return pi.detach().cpu().numpy(), v.detach().cpu().numpy()

    def stepNgetEmbedding(self,aigData,device='cuda'):
        """
        Returns policy and value estimates for given observations.
        :param obs: Array of shape [N] containing N observations.
        :return: Policy estimate [N, n_actions] and value estimate [N] for
        the given observations.
        """
        self.step_model.eval()
        loader = DataLoader(aigData, batch_size=self.batch_size)
        batchedAIG = next(iter(loader))
        batchedAIG = batchedAIG.to(device)
        with torch.no_grad():
            logits, pi,v,finalEmbed,aigEmbed = self.step_model(batchedAIG)
            return pi.detach().cpu().numpy(), v.detach().cpu().numpy(),aigEmbed.detach().cpu().numpy()
        
    def stepNgetLogits(self,aigData,device='cuda'):
        """
        Returns policy and value estimates for given observations.
        :param obs: Array of shape [N] containing N observations.
        :return: Policy estimate [N, n_actions] and value estimate [N] for
        the given observations.
        """
        self.step_model.eval()
        loader = DataLoader(aigData, batch_size=self.batch_size)
        batchedAIG = next(iter(loader))
        batchedAIG = batchedAIG.to(device)
        with torch.no_grad():
            logits, pi,v,finalEmbed,aigEmbed = self.step_model(batchedAIG)
            return pi.detach().cpu().numpy(), v.detach().cpu().numpy(),logits.detach().cpu().numpy()
        
        
    def stepNgetFullEmbedding(self,aigData,device='cuda'):
        """
        Returns policy and value estimates for given observations.
        :param obs: Array of shape [N] containing N observations.
        :return: Policy estimate [N, n_actions] and value estimate [N] for
        the given observations.
        """
        self.step_model.eval()
        loader = DataLoader(aigData, batch_size=self.batch_size)
        batchedAIG = next(iter(loader))
        batchedAIG = batchedAIG.to(device)
        with torch.no_grad():
            logits, pi,v,finalEmbed,aigEmbed = self.step_model(batchedAIG)
            return pi.detach().cpu().numpy(), v.detach().cpu().numpy(),finalEmbed.detach().cpu().numpy()
