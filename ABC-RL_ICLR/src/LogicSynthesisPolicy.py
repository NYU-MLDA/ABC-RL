import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing,GCNConv,GINConv,GATConv,TAGConv
from torch_geometric.nn import global_mean_pool, global_max_pool,SAGPooling,TopKPooling,ASAPooling,global_add_pool
from torch_geometric.nn.norm import BatchNorm,GraphNorm,LayerNorm,InstanceNorm
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.loader import DataLoader
from transformers import BertModel
from torch_sparse import SparseTensor
BERT_MODEL_NAME = 'bert-base-cased'
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
bert_model.to('cuda')
bert_model.eval()
            

allowable_features = {
    'node_type' : [0,1,2],
    'num_inverted_predecessors' : [0,1,2]
}

def get_node_feature_dims():
    return list(map(len, [
        allowable_features['node_type']
    ]))


full_node_feature_dims = get_node_feature_dims()


class NodeEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(NodeEncoder, self).__init__()
        self.node_type_embedding = torch.nn.Embedding(full_node_feature_dims[0], emb_dim)
        torch.nn.init.xavier_uniform_(self.node_type_embedding.weight.data)

    def forward(self, x):
        # First feature is node type, second feature is inverted predecessor
        x_embedding = self.node_type_embedding(x[:, 0])
        x_embedding = torch.cat((x_embedding, x[:,1].reshape(-1,1)), dim=1)
        #x_embedding = self.node_type_embedding(x)
        return x_embedding


class AIGEncoder(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self,node_encoder,input_dim,num_layer = 2,emb_dim = 128,gnn_type = 'gcn',norm_type='batch',final_layer_readout=True,pooling_type=None,pooling_ratio=0.8,readout_type=['max','sum']):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''
        super(AIGEncoder,self).__init__()
        self.num_layer = num_layer
        self.node_emb_size = input_dim
        self.node_encoder = node_encoder
        self.gnn_conv = GCNConv
        self.norm_type = BatchNorm
        self.isPooling = False if pooling_type == None else True
        self.pooling_ratio = pooling_ratio
        self.final_layer_readout = final_layer_readout
        
        
        ### Select the type of Graph Conv Networks
        if gnn_type == 'gin':
            self.gnn_conv = GINConv
        elif gnn_type == 'gat':
            self.gnn_conv = GATConv
        elif gnn_type == 'tag':
            self.gnn_conv = TAGConv
            
        ### Select the type of Normalization
        if norm_type == 'graph':
            self.norm_type = GraphNorm
        elif norm_type == 'layer':
            self.norm_type = LayerNorm
        elif norm_type == 'instance':
            self.norm_type = InstanceNorm
            
        ## Pooling Layers
        if pooling_type == 'topk':
           self.pool_type = TopKPooling
        elif pooling_type == 'sag':
            self.pool_type = SAGPooling
        elif pooling_type == 'asap':
            self.pool_type = ASAPooling

        ###List of GNNs and layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        if self.isPooling:
            self.pools = torch.nn.ModuleList()

        ## First layer
        self.convs.append(self.gnn_conv(input_dim, emb_dim))
        self.norms.append(self.norm_type(emb_dim))
        if self.isPooling:
            self.pools.append(self.pool_type(emb_dim))

        ## Intermediate Layers
        for _ in range(1, num_layer-1):
            self.convs.append(self.gnn_conv(emb_dim, emb_dim))
            self.norms.append(self.norm_type(emb_dim))
            if self.isPooling:
                self.pools.append(self.pool_type(in_channels=emb_dim,ratio=self.pooling_ratio))
            
        ## Last Layer
        self.convs.append(self.gnn_conv(emb_dim, emb_dim))
        self.norms.append(self.norm_type(emb_dim))
        
        
        ## Global Readout Layers
        self.readout = []
        for readoutConfig in readout_type:
            if readoutConfig == 'max':
                self.readout.append(global_max_pool)
            elif readoutConfig == 'mean':
                self.readout.append(global_mean_pool)
            elif readoutConfig == 'sum':
                self.readout.append(global_add_pool)
            


    def forward(self, batched_data):
        edge_index = batched_data.edge_index
        #adj_t = batched_data.adj_t
        batch = batched_data.batch
        adj = SparseTensor(row=edge_index[0], col=edge_index[1])

        x = torch.cat([batched_data.node_type.reshape(-1, 1),batched_data.num_inverted_predecessors.reshape(-1, 1)], dim=1)
        h = self.node_encoder(x)
        
        finalReadouts = []

        for layer in range(self.num_layer):
            #h = self.convs[layer](h, edge_index)
            h = self.convs[layer](h, adj.t())
            #h = self.norms[layer](h)
            #h = self.norms[layer](h)
            if layer != self.num_layer - 1:
                h = F.relu(h)
                if self.isPooling:                    # Not pooling in the last layer
                    poolOutput = self.pools[layer](h,edge_index=edge_index,batch=batch)
                    h,edge_index,batch = poolOutput[0],poolOutput[1],poolOutput[3]
                if self.final_layer_readout:
                    continue
            
            finalReadouts.append(self.readout[0](h,batch))
            finalReadouts.append(self.readout[1](h,batch))
        aigEmbedding = torch.cat(finalReadouts,dim=1)
        aigEmbedding = torch.round(aigEmbedding,decimals=3)
        return aigEmbedding


class LogicSynthesisPolicy(nn.Module):
    """
    Simple neural network policy for solving the hill climbing task.
    Consists of one common dense layer for both policy and value estimate and
    another dense layer for each.
    """
    def __init__(self, init_graph_data=True,node_enc_outdim=3, gnn_hidden_dim = 32,num_gcn_layer = 2,
                gnn_type = 'gcn',norm_type='batch',final_layer_readout=True,
                pooling_type=None,pooling_ratio=0.8,readout_type=['mean','max'], n_hidden=256,n_actions=7):
        
        super(LogicSynthesisPolicy, self).__init__()
        self.init_graph_data = init_graph_data
        if self.init_graph_data:
            self.node_encoder = NodeEncoder(emb_dim=node_enc_outdim)
            #Node encoding output has dimension 3 and number of incoming inverted edges has dimension 1
            self.aig = AIGEncoder(self.node_encoder,input_dim=node_enc_outdim+1,num_layer=num_gcn_layer,emb_dim=gnn_hidden_dim,gnn_type = gnn_type,
                                 norm_type=norm_type,final_layer_readout=final_layer_readout,pooling_type=pooling_type,pooling_ratio=pooling_ratio,readout_type=readout_type)
            
            #Readout happening after each GCN layer
            #Readouts can be multiple: max and mean
            self.aig_emb_dim = num_gcn_layer * gnn_hidden_dim * len(readout_type)
            if final_layer_readout == True:
               self.aig_emb_dim = gnn_hidden_dim*len(readout_type)
            self.aig_emb_dim+=768
        else:
            self.aig_emb_dim = 768
        
        self.n_hidden = n_hidden
        self.n_actions = n_actions

        self.denseLayer = nn.Linear(self.aig_emb_dim, n_hidden)
        self.dense_p1 = nn.Linear(n_hidden, n_hidden)
        self.dense_p2 = nn.Linear(n_hidden, n_actions)
        self.dense_v1 = nn.Linear(n_hidden, n_hidden)
        self.dense_v2 = nn.Linear(n_hidden, 1)
        torch.nn.init.xavier_uniform_(self.dense_p2.weight.data)
        torch.nn.init.kaiming_uniform_(self.dense_p1.weight.data)
        torch.nn.init.kaiming_uniform_(self.dense_v2.weight.data)
        torch.nn.init.kaiming_uniform_(self.dense_v1.weight.data)
        torch.nn.init.kaiming_uniform_(self.denseLayer.weight.data)
        # Google Brain: What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study
        # Multiply the last policy layer weights by 1e-2
        self.dense_p2.weight.data = self.dense_p2.weight.data*0.01 
        

    def forward(self, batchData):
        seqEmbedding = bert_model(batchData.data_input,batchData.data_attention)
        seqEmbedding = seqEmbedding.pooler_output
        if self.init_graph_data:
            init_aig_embedding = self.aig(batchData)
            finalEmbedding = torch.cat([init_aig_embedding,seqEmbedding],dim=1)
        else:
            finalEmbedding = seqEmbedding
        aigFCOutput = F.leaky_relu(self.denseLayer(finalEmbedding))
        p1Out = F.leaky_relu(self.dense_p1(aigFCOutput))
        v1Out = F.leaky_relu(self.dense_v1(aigFCOutput))
        logits = self.dense_p2(p1Out)
        policy = F.softmax(logits, dim=1)
        value = torch.tanh(self.dense_v2(v1Out)).view(-1)

        return logits, policy, value,finalEmbedding,init_aig_embedding
