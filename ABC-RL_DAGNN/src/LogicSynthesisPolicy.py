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

from dag_utils import subgraph, custom_backward_subgraph

from gat_conv import AGNNConv
from gcn_conv import AggConv
from deepset_conv import DeepSetConv
from gated_sum_conv import GatedSumConv

from torch.nn import LSTM, GRU

BERT_MODEL_NAME = 'bert-base-cased'
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
bert_model.to('cuda')
bert_model.eval()
            

_aggr_function_factory = {
    'aggnconv': AGNNConv,
    'deepset': DeepSetConv,
    'gated_sum': GatedSumConv,
    'conv_sum': AggConv,
}

_update_function_factory = {
    'lstm': LSTM,
    'gru': GRU,
}

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
    
    
class DAGNodeEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim=32):
        super(DAGNodeEncoder, self).__init__()
        self.node_type_embedding = torch.nn.Embedding(full_node_feature_dims[0], emb_dim)
        self.num_inverted_predecessor_embedding = torch.nn.Embedding(full_node_feature_dims[0], emb_dim)
        torch.nn.init.xavier_uniform_(self.node_type_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.num_inverted_predecessor_embedding.weight.data)

    def forward(self, x):
        # First feature is node type, second feature is inverted predecessor
        first_embedding = self.node_type_embedding(x[:, 0])
        second_embedding = self.num_inverted_predecessor_embedding(x[:, 1])
        x_embedding = torch.cat((first_embedding, second_embedding), dim=1)
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
        #aigEmbedding = torch.round(aigEmbedding,decimals=3)
        return aigEmbedding
    

class DAGConvGNN(nn.Module):
    '''
    Recurrent Graph Neural Networks for Circuits.
    '''
    def __init__(self,node_encoder,input_dim,num_rounds=1,device='cuda',reverse=True,custom_backward=False,use_edge_attr=True,num_aggr=3,dim_hidden=32,num_fc=3,readout_type=['max','sum']):
        super(DAGConvGNN, self).__init__()
        
         # configuration
        self.num_rounds = num_rounds
        self.device = device
        self.reverse = reverse
        self.custom_backward = custom_backward
        self.use_edge_attr = use_edge_attr

        # dimensions
        self.node_encoder = node_encoder
        self.num_aggr = num_aggr
        self.dim_node_feature = input_dim
        self.dim_hidden = dim_hidden
        self.num_fc = num_fc
        self.aggr_function = 'conv_sum'
        self.update_function = 'gru'
        self.wx_update = False
        
        ## Global Readout Layers
        self.readout = []
        for readoutConfig in readout_type:
            if readoutConfig == 'max':
                self.readout.append(global_max_pool)
            elif readoutConfig == 'mean':
                self.readout.append(global_mean_pool)
            elif readoutConfig == 'sum':
                self.readout.append(global_add_pool)

        # 1. message/aggr-related
        dim_aggr = self.dim_node_feature #self.dim_hidden# + self.dim_edge_feature if self.use_edge_attr else self.dim_hidden
        if self.aggr_function in _aggr_function_factory.keys():
            # if self.use_edge_attr:
            #     aggr_forward_pre = MLP(self.dim_hidden, self.dim_hidden, self.dim_hidden, num_layer=3, p_drop=0.2)
            # else:
            aggr_forward_pre = nn.Linear(dim_aggr, self.dim_hidden)
            if self.aggr_function == 'deepset':
                aggr_forward_post = nn.Linear(self.dim_hidden, self.dim_hidden)
                self.aggr_forward = _aggr_function_factory[self.aggr_function](dim_aggr, self.dim_hidden, mlp=aggr_forward_pre, mlp_post=aggr_forward_post, wea=self.use_edge_attr)
            else:
                self.aggr_forward = _aggr_function_factory[self.aggr_function](dim_aggr, self.dim_hidden, mlp=aggr_forward_pre, wea=self.use_edge_attr)
            if self.reverse:
                # if self.use_edge_attr:
                #     aggr_backward_pre = MLP(self.dim_hidden, self.dim_hidden, self.dim_hidden, num_layer=3, p_drop=0.2)
                # else:
                aggr_backward_pre = nn.Linear(dim_aggr, self.dim_hidden)
                if self.aggr_function == 'deepset':
                    aggr_backward_post = nn.Linear(self.dim_hidden, self.dim_hidden)
                    self.aggr_backward = _aggr_function_factory[self.aggr_function](dim_aggr, self.dim_hidden, mlp=aggr_backward_pre, mlp_post=aggr_backward_post, wea=self.use_edge_attr)
                else:
                    self.aggr_backward = _aggr_function_factory[self.aggr_function](dim_aggr, self.dim_hidden, mlp=aggr_backward_pre, reverse=True, wea=self.use_edge_attr)
        else:
            raise KeyError('no support {} aggr function.'.format(self.aggr_function))


        # 2. update-related
        if self.update_function in _update_function_factory.keys():
            # Here only consider the inputs as the concatenated vector from embedding and feature vector.
            if self.wx_update:
                self.update_forward = _update_function_factory[self.update_function](self.dim_node_feature+self.dim_hidden, self.dim_hidden)
                if self.reverse:
                    self.update_backward = _update_function_factory[self.update_function](self.dim_node_feature+self.dim_hidden, self.dim_hidden)
            else:
                #self.update_forward = _update_function_factory[self.update_function](self.dim_hidden, self.dim_hidden)
                self.update_forward = _update_function_factory[self.update_function](self.dim_hidden, self.dim_node_feature)
                if self.reverse:
                    #self.update_backward = _update_function_factory[self.update_function](self.dim_hidden, self.dim_hidden)
                    self.update_backward = _update_function_factory[self.update_function](self.dim_hidden,  self.dim_node_feature)
        else:
            raise KeyError('no support {} update function.'.format(self.update_function))
        # consider the embedding for the LSTM/GRU model initialized by non-zeros
        self.one = torch.ones(1).to(self.device)
        self.emd_int = nn.Linear(1, self.dim_hidden)
        self.one.requires_grad = False

        

    def forward(self, G):
        num_nodes = G.nodes
        num_layers_f = max(G.forward_level).item() + 1
        num_layers_b = max(G.backward_level).item() + 1
        #one = self.one
        #h_init = self.emd_int(one).view(1, 1, -1) # (1 x 1 x dim_hidden)
        #h_init = h_init.repeat(1, num_nodes, 1) # (1 x num_nodes x dim_hidden)
        edge_index = G.edge_index
        #adj_t = batched_data.adj_t
        #batch = G.batch
        #adj = SparseTensor(row=edge_index[0], col=edge_index[1])

        x = torch.cat([G.node_type.reshape(-1, 1),G.num_inverted_predecessors.reshape(-1, 1)], dim=1)
        h_init = self.node_encoder(x)
        h_init = torch.unsqueeze(h_init,dim=0)
        
        if self.update_function == 'lstm':
            node_embedding = self._lstm_forward(G, h_init, num_layers_f, num_layers_b, num_nodes)
        elif self.update_function == 'gru':
            node_embedding = self._gru_forward(G, h_init, num_layers_f, num_layers_b, num_nodes)
        else:
            raise NotImplementedError('The update function should be specified as one of lstm and gru.')
        
        return node_embedding
    
    def _lstm_forward(self, G, h_init, num_layers_f, num_layers_b, num_nodes):
        #x, edge_index = G.x, G.edge_index
        edge_index = G.edge_index
        #edge_attr = G.edge_attr if self.use_edge_attr else None
        edge_attr = G.edge_attr if self.use_edge_attr else None
        
        node_state = (h_init, torch.zeros(1, num_nodes, self.dim_hidden).to(self.device)) # (h_0, c_0). here we only initialize h_0. TODO: option of not initializing the hidden state of LSTM.
        
        # TODO: add supports for modified attention and customized backward design.
        preds = []
        for _ in range(self.num_rounds):
            for l_idx in range(1, num_layers_f):
                # forward layer
                layer_mask = G.forward_level == l_idx
                l_node = G.forward_index[layer_mask]

                l_state = (torch.index_select(node_state[0], dim=1, index=l_node), 
                            torch.index_select(node_state[1], dim=1, index=l_node))

                l_edge_index, l_edge_attr = subgraph(l_node, edge_index, edge_attr, dim=1)
                msg = self.aggr_forward(node_state[0].squeeze(0), l_edge_index, l_edge_attr)
                l_msg = torch.index_select(msg, dim=0, index=l_node)
                #l_x = torch.index_select(x, dim=0, index=l_node)
                
                # if self.wx_update:
                #     _, l_state = self.update_forward(torch.cat([l_msg, l_x], dim=1).unsqueeze(0), l_state)
                # else:
                #     _, l_state = self.update_forward(l_msg.unsqueeze(0), l_state)
                
                _, l_state = self.update_forward(l_msg.unsqueeze(0), l_state)
                node_state[0][:, l_node, :] = l_state[0]
                node_state[1][:, l_node, :] = l_state[1]
            if self.reverse:
                for l_idx in range(1, num_layers_b):
                    # backward layer
                    layer_mask = G.backward_level == l_idx
                    l_node = G.backward_index[layer_mask]
                    
                    l_state = (torch.index_select(node_state[0], dim=1, index=l_node), 
                                torch.index_select(node_state[1], dim=1, index=l_node))
                    if self.custom_backward:
                        l_edge_index = custom_backward_subgraph(l_node, edge_index, device=self.device, dim=0)
                    else:
                        l_edge_index, l_edge_attr = subgraph(l_node, edge_index, edge_attr, dim=0)
                    msg = self.aggr_backward(node_state[0].squeeze(0), l_edge_index, l_edge_attr)
                    l_msg = torch.index_select(msg, dim=0, index=l_node)
                    #l_x = torch.index_select(x, dim=0, index=l_node)
                    
                    # if self.wx_update:
                    #     _, l_state = self.update_backward(torch.cat([l_msg, l_x], dim=1).unsqueeze(0), l_state)
                    # else:
                    #     _, l_state = self.update_backward(l_msg.unsqueeze(0), l_state)
                    
                    _, l_state = self.update_backward(l_msg.unsqueeze(0), l_state)                    
                    node_state[0][:, l_node, :] = l_state[0]
                    node_state[1][:, l_node, :] = l_state[1]
               
            
        node_embedding = node_state[0].squeeze(0)

        return node_embedding
    
    
    def _get_output_nodes(self, G):
        output_mask = G.node_type == 1
        output_nodes_index = G.forward_index[output_mask]
        return output_nodes_index
        
    
    def _gru_forward(self, G, h_init, num_layers_f, num_layers_b, num_nodes):
        #x, edge_index = G.x, G.edge_index
        edge_index = G.edge_index
        #edge_attr = G.edge_attr if self.use_edge_attr else None
        edge_attr = G.edge_attr if self.use_edge_attr else None

        node_state = h_init # (h_0). here we initialize h_0. TODO: option of not initializing the hidden state of GRU.

        # TODO: add supports for modified attention and customized backward design.
        preds = []
        for _ in range(self.num_rounds):
            for l_idx in range(1, num_layers_f):
                # forward layer
                layer_mask = G.forward_level == l_idx
                l_node = G.forward_index[layer_mask]
                #print(node_state.size())
                #print(l_node)
                l_state = torch.index_select(node_state, dim=1, index=l_node)
                #print(edge_index.size())
                l_edge_index, l_edge_attr = subgraph(l_node, edge_index, edge_attr, dim=1)
                #print(l_edge_index.size())
                msg = self.aggr_forward(node_state.squeeze(0), l_edge_index, l_edge_attr)
                l_msg = torch.index_select(msg, dim=0, index=l_node)
                #l_x = torch.index_select(x, dim=0, index=l_node)
                
                # if self.wx_update:
                #     _, l_state = self.update_forward(torch.cat([l_msg, l_x], dim=1).unsqueeze(0), l_state)
                # else:
                #     _, l_state = self.update_forward(l_msg.unsqueeze(0), l_state)
                #print(l_msg.unsqueeze(0).size())
                #_, l_state = self.update_forward(l_msg.unsqueeze(0), l_state)
                #print(l_msg.size())
                #print(node_state.size())
                node_state[:, l_node, :] = l_msg
            
            if self.reverse:
                for l_idx in range(1, num_layers_b):
                    # backward layer
                    layer_mask = G.backward_level == l_idx
                    l_node = G.backward_index[layer_mask]
                    
                    l_state = torch.index_select(node_state, dim=1, index=l_node)

                    if self.custom_backward:
                        l_edge_index = custom_backward_subgraph(l_node, edge_index, device=self.device, dim=0)
                    else:
                        l_edge_index, l_edge_attr = subgraph(l_node, edge_index, edge_attr, dim=0)
                    msg = self.aggr_backward(node_state.squeeze(0), l_edge_index, l_edge_attr)
                    l_msg = torch.index_select(msg, dim=0, index=l_node)
                    #l_x = torch.index_select(x, dim=0, index=l_node)
                    
                    # if self.wx_update:
                    #     _, l_state = self.update_backward(torch.cat([l_msg, l_x], dim=1).unsqueeze(0), l_state)
                    # else:
                    #     _, l_state = self.update_backward(l_msg.unsqueeze(0), l_state)    
                    _, l_state = self.update_backward(l_msg.unsqueeze(0), l_state)            
                    node_state[:, l_node, :] = l_msg

        node_embedding = node_state.squeeze(0)
        #print(node_embedding.size())
        
        finalReadouts = []

        output_node_indices = self._get_output_nodes(G)
        finalReadouts.append(self.readout[0](node_embedding[output_node_indices],G.batch[output_node_indices]))
        finalReadouts.append(self.readout[1](node_embedding[output_node_indices],G.batch[output_node_indices]))
        aigEmbedding = torch.cat(finalReadouts,dim=1)
        #aigEmbedding = torch.round(aigEmbedding,decimals=4)
        #print(aigEmbedding.size())
        return aigEmbedding

        #return node_embedding


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
            #self.node_encoder = NodeEncoder(emb_dim=node_enc_outdim)
            #Node encoding output has dimension 3 and number of incoming inverted edges has dimension 1
            #self.aig = AIGEncoder(self.node_encoder,input_dim=node_enc_outdim+1,num_layer=num_gcn_layer,emb_dim=gnn_hidden_dim,gnn_type = gnn_type,
            #                    norm_type=norm_type,final_layer_readout=final_layer_readout,pooling_type=pooling_type,pooling_ratio=pooling_ratio,readout_type=readout_type)
            
            self.dag_node_encoder = DAGNodeEncoder(emb_dim=16)
            self.aig = DAGConvGNN(self.dag_node_encoder,input_dim=32,num_rounds=1,device='cuda',reverse=False,custom_backward=False,use_edge_attr=False,num_aggr=3,dim_hidden=32,num_fc=3,readout_type=readout_type)
            
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
            # finalEmbedding = init_aig_embedding
        else:
            finalEmbedding = seqEmbedding
        print("init graph data status",self.init_graph_data)
        print(finalEmbedding.size())
        aigFCOutput = F.leaky_relu(self.denseLayer(finalEmbedding))
        p1Out = F.leaky_relu(self.dense_p1(aigFCOutput))
        v1Out = F.leaky_relu(self.dense_v1(aigFCOutput))
        logits = self.dense_p2(p1Out)
        policy = F.softmax(logits, dim=1)
        value = torch.tanh(self.dense_v2(v1Out)).view(-1)

        return logits, policy, value,finalEmbedding,init_aig_embedding


# class LogicSynthesisPolicy(nn.Module):
#     """
#     Simple neural network policy for solving the hill climbing task.
#     Consists of one common dense layer for both policy and value estimate and
#     another dense layer for each.
#     """
#     def __init__(self, init_graph_data=True,node_enc_outdim=3, gnn_hidden_dim = 32,num_gcn_layer = 2,
#                 gnn_type = 'gcn',norm_type='batch',final_layer_readout=True,
#                 pooling_type=None,pooling_ratio=0.8,readout_type=['mean','max'], n_hidden=256,n_actions=7):
        
#     #node_encoder,input_dim,num_layer = 2,emb_dim = 128,gnn_type = 'gcn',norm_type='batch',final_layer_readout=True,pooling_type=None,pooling_ratio=0.8,readout_type=['max','sum']
        
#         super(LogicSynthesisPolicy, self).__init__()
#         #self.aig = AIGEncoder(self.node_encoder,input_dim=node_enc_outdim+1,num_layer=num_gcn_layer,emb_dim=gnn_hidden_dim,gnn_type = gnn_type,
# #       #                         norm_type=norm_type,final_layer_readout=final_layer_readout,pooling_type=pooling_type,pooling_ratio=pooling_ratio,readout_type=readout_type)
#         self.node_encoder = NodeEncoder(emb_dim=node_enc_outdim)
#         self.num_layer = num_gcn_layer
#         self.emb_dim = gnn_hidden_dim
#         self.node_emb_size = node_enc_outdim+1
#         self.gnn_conv = GCNConv
#         self.norm_type = BatchNorm
#         self.isPooling = False 
#         self.pooling_ratio = pooling_ratio
#         self.final_layer_readout = final_layer_readout
#         self.input_dim=node_enc_outdim+1
        
        
#         ### Select the type of Graph Conv Networks
#         if gnn_type == 'gin':
#             self.gnn_conv = GINConv
#         elif gnn_type == 'gat':
#             self.gnn_conv = GATConv
#         elif gnn_type == 'tag':
#             self.gnn_conv = TAGConv
            
#         ### Select the type of Normalization
#         if norm_type == 'graph':
#             self.norm_type = GraphNorm
#         elif norm_type == 'layer':
#             self.norm_type = LayerNorm
#         elif norm_type == 'instance':
#             self.norm_type = InstanceNorm
            
#         ## Pooling Layers
#         if pooling_type == 'topk':
#            self.pool_type = TopKPooling
#         elif pooling_type == 'sag':
#             self.pool_type = SAGPooling
#         elif pooling_type == 'asap':
#             self.pool_type = ASAPooling

#         ###List of GNNs and layers
#         self.convs = torch.nn.ModuleList()
#         self.norms = torch.nn.ModuleList()
#         if self.isPooling:
#             self.pools = torch.nn.ModuleList()

#         ## First layer
#         self.convs.append(self.gnn_conv(self.input_dim, self.emb_dim))
#         self.norms.append(self.norm_type(self.emb_dim))
#         if self.isPooling:
#             self.pools.append(self.pool_type(self.emb_dim))

#         ## Intermediate Layers
#         for _ in range(1, self.num_layer-1):
#             self.convs.append(self.gnn_conv(self.emb_dim, self.emb_dim))
#             self.norms.append(self.norm_type(self.emb_dim))
#             if self.isPooling:
#                 self.pools.append(self.pool_type(in_channels=self.emb_dim,ratio=self.pooling_ratio))
            
#         ## Last Layer
#         self.convs.append(self.gnn_conv(self.emb_dim, self.emb_dim))
#         self.norms.append(self.norm_type(self.emb_dim))
        
        
#         ## Global Readout Layers
#         self.readout = []
#         for readoutConfig in readout_type:
#             if readoutConfig == 'max':
#                 self.readout.append(global_max_pool)
#             elif readoutConfig == 'mean':
#                 self.readout.append(global_mean_pool)
#             elif readoutConfig == 'sum':
#                 self.readout.append(global_add_pool)
        
#         self.init_graph_data = init_graph_data
#         if self.init_graph_data:
#             self.node_encoder = NodeEncoder(emb_dim=node_enc_outdim)
#             #Node encoding output has dimension 3 and number of incoming inverted edges has dimension 1
#             self.aig = AIGEncoder(self.node_encoder,input_dim=node_enc_outdim+1,num_layer=num_gcn_layer,emb_dim=gnn_hidden_dim,gnn_type = gnn_type,
#                                 norm_type=norm_type,final_layer_readout=final_layer_readout,pooling_type=pooling_type,pooling_ratio=pooling_ratio,readout_type=readout_type)
            
#             #self.dag_node_encoder = DAGNodeEncoder(emb_dim=16)
#             #self.aig = DAGConvGNN(self.dag_node_encoder,input_dim=32,num_rounds=1,device='cuda',reverse=False,custom_backward=False,use_edge_attr=False,num_aggr=3,dim_hidden=32,num_fc=3,readout_type=readout_type)
            
#             #Readout happening after each GCN layer
#             #Readouts can be multiple: max and mean
#             self.aig_emb_dim = num_gcn_layer * gnn_hidden_dim * len(readout_type)
#             if final_layer_readout == True:
#                self.aig_emb_dim = gnn_hidden_dim*len(readout_type)
#             #self.aig_emb_dim+=768
#         else:
#             self.aig_emb_dim = 768
        
#         self.n_hidden = n_hidden
#         self.n_actions = n_actions

#         self.denseLayer = nn.Linear(self.aig_emb_dim, n_hidden)
#         self.denseLayer1 = nn.Linear(self.aig_emb_dim, n_actions)
#         self.dense_p1 = nn.Linear(n_hidden, n_hidden)
#         self.dense_p2 = nn.Linear(n_hidden, n_actions)
#         self.dense_v1 = nn.Linear(n_hidden, n_hidden)
#         self.dense_v2 = nn.Linear(n_hidden, 1)
#         torch.nn.init.xavier_uniform_(self.dense_p2.weight.data)
#         torch.nn.init.kaiming_uniform_(self.dense_p1.weight.data)
#         torch.nn.init.kaiming_uniform_(self.dense_v2.weight.data)
#         torch.nn.init.kaiming_uniform_(self.dense_v1.weight.data)
#         torch.nn.init.kaiming_uniform_(self.denseLayer.weight.data)
#         torch.nn.init.kaiming_uniform_(self.denseLayer1.weight.data)
#         # Google Brain: What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study
#         # Multiply the last policy layer weights by 1e-2
#         self.dense_p2.weight.data = self.dense_p2.weight.data*0.01 
        

#     def forward(self, batchData):
#         #seqEmbedding = bert_model(batchData.data_input,batchData.data_attention)
#         #seqEmbedding = seqEmbedding.pooler_output
#         if self.init_graph_data:
#             edge_index = batchData.edge_index
#             #adj_t = batched_data.adj_t
#             batch = batchData.batch
#             #adj = SparseTensor(row=edge_index[0], col=edge_index[1])

#             x = torch.cat([batchData.node_type.reshape(-1, 1),batchData.num_inverted_predecessors.reshape(-1, 1)], dim=1)
#             h = self.node_encoder(x)
            
#             finalReadouts = []
            
            
#             h = F.relu(self.norms[0](self.convs[0](h, edge_index)))
#             h = self.norms[1](self.convs[1](h, edge_index))

#             finalReadouts.append(self.readout[0](h,batch))
#             finalReadouts.append(self.readout[1](h,batch))
                
#             aigEmbedding = torch.cat(finalReadouts,dim=1)
#             #aigEmbedding = torch.cat([global_mean_pool(h,batch),global_max_pool(h,batch)],dim=1)
#             aigEmbedding = torch.round(aigEmbedding,decimals=3)
#             #aigEmbedding = F.dropout(aigEmbedding, p=0.5, training=self.training)
#             #init_aig_embedding = self.aig(batchData)
#             #finalEmbedding = torch.cat([init_aig_embedding,seqEmbedding],dim=1)
#             finalEmbedding = aigEmbedding
#             init_aig_embedding = aigEmbedding
#         # else:
#         #     finalEmbedding = seqEmbedding
#         print("init graph data status",self.init_graph_data)
#         print(finalEmbedding.size())
#         aigFCOutput = F.leaky_relu(self.denseLayer(finalEmbedding))
#         p1Out = F.leaky_relu(self.dense_p1(aigFCOutput))
#         v1Out = F.leaky_relu(self.dense_v1(aigFCOutput))
#         #logits = self.dense_p2(p1Out)
#         logits = self.denseLayer1(finalEmbedding)
#         policy = F.softmax(logits, dim=1)
#         value = torch.tanh(self.dense_v2(v1Out)).view(-1)

#         return logits, policy, value,finalEmbedding,init_aig_embedding
