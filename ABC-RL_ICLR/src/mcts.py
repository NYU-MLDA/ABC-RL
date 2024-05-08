"""
Adapted from https://github.com/tensorflow/minigo/blob/master/mcts.py

Implementation of the Monte-Carlo tree search algorithm as detailed in the
AlphaGo Zero paper (https://www.nature.com/articles/nature24270).
"""
import math
import random as rd
import collections
from unicodedata import name
import numpy as np
import os
from zipfile import ZipFile
import os.path as osp
import torch
import zipfile

# Exploration constant
c_PUCT = 1.38
# Dirichlet noise alpha parameter.
D_NOISE_ALPHA = 0.03
# Number of steps into the episode after which we always select the
# action with highest action probability rather than selecting randomly
TEMP_THRESHOLD = 5
CHILD_TYPE = 'robust'


class DummyNode:
    """
    Special node that is used as the node above the initial root node to
    prevent having to deal with special cases when traversing the tree.
    """

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)
        self.child_M = collections.defaultdict(float)

    def revert_visits(self, up_to=None): pass

    def backup_value(self, predValue,actualValue,up_to=None): pass


class MCTSNode:
    """
    Represents a node in the Monte-Carlo search tree. Each node holds a single
    environment state.
    """

    def __init__(self,state,ptstate,n_actions, TreeEnv, action=None, parent=None,graphData=None,childType='robust'):
        """
        :param state: State that the node should hold.
        :param n_actions: Number of actions that can be performed in each
        state. Equal to the number of outgoing edges of the node.
        :param TreeEnv: Static class that defines the environment dynamics,
        e.g. which state follows from another state when performing an action.
        :param action: Index of the action that led from the parent node to
        this node.
        :param parent: Parent node.
        """
        self.TreeEnv = TreeEnv
        if parent is None:
            self.depth = 0
            parent = DummyNode()
        else:
            self.depth = parent.depth+1
        self.parent = parent
        self.action = action
        self.state = state
        self.ptstate = ptstate
        self.n_actions = n_actions
        self.is_expanded = False
        self.childType = childType
        self.graphData = graphData
        self.n_vlosses = 0  # Number of virtual losses on this node
        self.child_N = np.zeros([n_actions], dtype=np.float32)
        self.child_W = np.zeros([n_actions], dtype=np.float32)
        #self.child_W = np.ones([n_actions], dtype=np.float32)*(-1.0)
        self.child_M = np.zeros([n_actions], dtype=np.float32)
        #self.child_M = np.ones([n_actions], dtype=np.float32)*(-1.0)
        # Save copy of original prior before it gets mutated by dirichlet noise
        self.original_prior = np.zeros([n_actions], dtype=np.float32)
        self.child_prior = np.zeros([n_actions], dtype=np.float32)
        self.child_visited = np.zeros([n_actions], dtype=np.int32)
        self.children = {}

    @property
    def N(self):
        """
        Returns the current visit count of the node.
        """
        return self.parent.child_N[self.action]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.action] = value

    @property
    def W(self):
        """
        Returns the current total value of the node.
        """
        return self.parent.child_W[self.action]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.action] = value
        
    @property
    def M(self):
        """
        Returns the current total value of the node.
        """
        return self.parent.child_M[self.action]

    @M.setter
    def M(self, value):
        self.parent.child_M[self.action] = value

    @property
    def Q(self):
        """
        Returns the current action value of the node.
        """
        return self.W / (1 + self.N)
    
    @property
    def averagedMonteCarlo(self):
        """
        Returns the averaged montecarlo value of the node.
        """
        return self.M / (1 + self.N)
    
    @property
    def child_averagedMonteCarlo(self):
        return self.child_M / (1 + self.child_N)

    @property
    def child_Q(self):
        return self.child_W / (1 + self.child_N)

    @property
    def child_U(self):
        return (c_PUCT * math.sqrt(1 + self.N) *
                self.child_prior / (1 + self.child_N))

    @property
    def child_action_score(self):
        """
        Action_Score(s, a) = Q(s, a) + U(s, a) as in paper. A high value
        means the node should be traversed.
        """
        #return self.child_Q + self.child_U
        return self.child_averagedMonteCarlo + self.child_U
    
    
    def select_leaf(self):
        """
        Traverses the MCT rooted in the current node until it finds a leaf
        (i.e. a node that only exists in its parent node in terms of its
        child_N and child_W values but not as a dedicated node in the parent's
        children-mapping). Nodes are selected according to child_action_score.
        It expands the leaf by adding a dedicated MCTSNode. Note that the
        estimated value and prior probabilities still have to be set with
        `incorporate_estimates` afterwards.
        :return: Expanded leaf MCTSNode.
        """
        current = self
        while True:
            #current.N += 1
            # Encountered leaf node (i.e. node that is not yet expanded).
            if not current.is_expanded:
                break
            # Logic: For an expanded node, explore all child at least once. 
            # Once, all the child are added as leaf and averaged monte carlo
            # rollout has been evaluated, choose the action with highest score.
            # if np.sum(current.child_visited) < current.n_actions:
            #     for i in range(len(current.child_visited)):
            #         if current.child_visited[i] == 0:
            #             best_move = i
            #             break
            # else:
            #     best_move = np.argmax(current.child_action_score)
            best_move = np.argmax(current.child_action_score)
            current = current.maybe_add_child(best_move)
        return current

    def exploit_leaf(self):
        """
        Traverses the MCT rooted in the current node until it finds a leaf
        (i.e. a node that only exists in its parent node in terms of its
        child_N and child_W values but not as a dedicated node in the parent's
        children-mapping). Nodes are selected according to child_action_score.
        It expands the leaf by adding a dedicated MCTSNode. Note that the
        estimated value and prior probabilities still have to be set with
        `incorporate_estimates` afterwards.
        :return: Expanded leaf MCTSNode.
        """
        current = self
        while True:
            #current.N += 1
            # Encountered leaf node (i.e. node that is not yet expanded).
            if not current.is_expanded:
                break
            # Logic: For an expanded node, explore all child at least once. 
            # Once, all the child are added as leaf and averaged monte carlo
            # rollout has been evaluated, choose the action with highest score.
            # if np.sum(current.child_visited) < current.n_actions:
            #     for i in range(len(current.child_visited)):
            #         if current.child_visited[i] == 0:
            #             best_move = i
            #             break
            # else:
            #     best_move = np.argmax(current.child_action_score)
            if self.childType == "robust":
                best_move = np.argmax(current.child_N)
            else:
                best_move = np.argmax(current.averagedMonteCarlo)
            current = current.maybe_add_child(best_move)
        return current

    def maybe_add_child(self, action):
        """
        Adds a child node for the given action if it does not yet exists, and
        returns it.
        :param action: Action to take in current state which leads to desired
        child node.
        :return: Child MCTSNode.
        """
        if action not in self.children:
            # Obtain state following given action.
            new_state = self.TreeEnv.next_state(self.state,self.depth,action)
            new_pt_state = os.path.splitext(new_state)[0]+'.pt.zip'
            self.children[action] = MCTSNode(new_state, new_pt_state,self.n_actions,
                                             self.TreeEnv,
                                             action=action, parent=self,graphData=self.graphData,childType=self.childType)
            self.child_visited[action] = 1
        return self.children[action]
        
    def incorporate_estimates(self,action_probs,predValue,actualValue,up_to):
        """
        Call if the node has just been expanded via `select_leaf` to
        incorporate the prior action probabilities and state value estimated
        by the neural network.
        :param action_probs: Action probabilities for the current node's state
        predicted by the neural network.
        :param value: Value of the current node's state predicted by the neural
        network.
        :param up_to: The node to propagate until.
        """
        # A done node (i.e. episode end) should not go through this code path.
        # Rather it should directly call `backup_value` on the final node.
        # TODO: Add assert here
        # Another thread already expanded this node in the meantime.
        # Ignore wasted computation but correct visit counts.
        self.is_expanded = True
        self.original_prior = self.child_prior = action_probs
        # This is a deviation from the paper that led to better results in
        # practice (following the MiniGo implementation).
        #self.child_W = np.ones([self.n_actions], dtype=np.float32) * predValue
        #self.child_M = np.ones([self.n_actions], dtype=np.float32) * actualValue
        self.child_W = np.zeros([self.n_actions], dtype=np.float32)
        self.child_M = np.zeros([self.n_actions], dtype=np.float32)
        self.backup_value(predValue,actualValue,up_to=up_to)

    def backup_value(self, predValue,actualValue, up_to):
        """
        Propagates a value estimation up to the root node.
        :param value: Value estimate to be propagated.
        :param up_to: The node to propagate until.
        """
        self.W += predValue
        self.M += actualValue
        self.N += 1
        if self.parent is None or self is up_to:
            return
        self.parent.backup_value(predValue,actualValue,up_to)

    def is_done(self):
        return self.TreeEnv.is_done_state(self.depth)

    def inject_noise(self):
        dirch = np.random.dirichlet([D_NOISE_ALPHA] * self.n_actions)
        #self.child_prior = self.child_prior * 0.75 + dirch * 0.25
        self.child_prior = self.child_prior * 0.85 + dirch * 0.15

    def visits_as_probs(self, squash=False):
        """
        Returns the child visit counts as a probability distribution.
        :param squash: If True, exponentiate the probabilities by a temperature
        slightly large than 1 to encourage diversity in early steps.
        :return: Numpy array of shape (n_actions).
        """
        if self.childType == 'max':
            #probs = self.child_Q # MCTS max child
            probs = self.child_averagedMonteCarlo # MCTS max child
        else:
            probs = self.child_N # MCTS robust child
        if squash:
            probs = probs ** .95
        return probs / np.sum(probs)
    
    def print_bfs_tree(self, level=0):
        self.printNodeInfo(level)
        listOfNodesToPrint = []
        for _, child in sorted(self.children.items()):
            listOfNodesToPrint.append((child,level+1))
        while len(listOfNodesToPrint)>0:
            childNode,depth = listOfNodesToPrint.pop(0)
            childNode.printNodeInfo(depth)
            for _, child in sorted(childNode.children.items()):
                listOfNodesToPrint.append((child,depth+1))
            
    def printNodeInfo(self,level):
        #node_string = "\033[94m|" + "----"*level
        node_string = "----"
        node_string += "\n Tree depth: {}".format(level)
        node_string += "\n Node: action={}".format(self.action)
        node_string += "\n• state:{}".format(os.path.split(self.state)[1])
        node_string += "\n• Child averaged monte carlo:{}".format(self.child_averagedMonteCarlo)
        node_string += "\n• Child visitation:{}".format(self.child_N)
        node_string += "\n• Child probablities:{}".format(self.child_prior)
        node_string += "\n• Child Action scores:{}".format(self.child_action_score)
        node_string += "\n• N={},Q={},M={}".format(self.N,self.Q,self.averagedMonteCarlo)
        print(node_string)

    def print_tree(self, level=0):
        node_string = "\033[94m|" + "----"*level
        node_string += "\n Node: action={}\033[0m".format(self.action)
        node_string += "\n• state:\n{}".format(self.state)
        node_string += "\n• Child Action scores:\n{}".format(self.child_action_score)
        node_string += "\n• Child visitation:\n{}".format(self.child_visited)
        node_string += "\n• N={},Q={},M={}".format(self.N,self.Q,self.averagedMonteCarlo)
        print(node_string)
        for _, child in sorted(self.children.items()):
            child.print_tree(level+1)


class MCTS:
    """
    Represents a Monte-Carlo search tree and provides methods for performing
    the tree search.
    """

    def __init__(self, agent_netw, TreeEnv, childType='robust',
                 simulations_per_move=800, num_parallel=4,alphaValue=1):
        """
        :param agent_netw: Network for predicting action probabilities and
        state value estimate.
        :param TreeEnv: Static class that defines the environment dynamics,
        e.g. which state follows from another state when performing an action.
        :param seconds_per_move: Currently unused.
        :param simulations_per_move: Number of traversals through the tree
        before performing a step.
        :param num_parallel: Number of leaf nodes to collect before evaluating
        them in conjunction.
        """
        self.agent_netw = agent_netw
        self.TreeEnv = TreeEnv
        self.childType = childType
        self.simulations_per_move = simulations_per_move
        self.num_parallel = num_parallel
        self.temp_threshold = None        # Overwritten in initialize_search
        self.isFirstExploration = True
        self.alphaValue=alphaValue

        self.qs = []
        self.mc = []
        self.searches_pi = []
        self.obs = []

        self.root = None

    def initialize_search(self, state=None):
        init_state = self.TreeEnv.initial_state()
        n_actions = self.TreeEnv.n_actions
        init_pt_state = os.path.splitext(init_state)[0]+'.pt.zip'
        init_graph_data = self.TreeEnv.extract_init_graph(init_state)
        self.root = MCTSNode(init_state,init_pt_state,n_actions, self.TreeEnv,graphData=init_graph_data,childType=self.childType)
        # Number of steps into the episode after which we always select the
        # action with highest action probability rather than selecting randomly
        self.temp_threshold = TEMP_THRESHOLD
        self.qs = []
        self.mc = []
        self.searches_pi = []
        self.obs = []

    def getPytorchDataOfIntermediateAIGs(self,leaves):
        graphDataList = []
        for leaf in leaves:
            if os.path.exists(leaf.ptstate):
                filePathName = osp.basename(osp.splitext(leaf.ptstate)[0])
                with ZipFile(leaf.ptstate) as myzip:
                    with myzip.open(filePathName) as myfile:
                        data = torch.load(myfile)
            else:
                data = self.TreeEnv.extract_pytorch_graph(leaf.state,leaf.graphData)
                ptFilePath = leaf.ptstate.split('.zip')[0]
                torch.save(data,ptFilePath)
                with zipfile.ZipFile(leaf.ptstate,'w',zipfile.ZIP_DEFLATED) as fzip:
                    fzip.write(ptFilePath,arcname=osp.basename(ptFilePath))
                os.remove(ptFilePath)
            graphDataList.append(data)
        return graphDataList
        
    def tree_search(self):       
        leaf = self.root.select_leaf()
        if leaf.is_done():
            value = self.TreeEnv.get_return(leaf.state)
            leaf.backup_value(value,value,up_to=self.root)
        else:
            graphState = self.getPytorchDataOfIntermediateAIGs([leaf])
            action_probs, values = self.agent_netw.step(graphState)
            ## Compute montecarlo return using the policy's best move rather than random policy
            startingAction = np.argmax(action_probs[0])
            rolloutReturn = self.TreeEnv.get_montecarlo_return(startingAction,leaf.state,leaf.depth)
            #leaf.incorporate_estimates(action_probs[0],values[0],rolloutReturn,up_to=self.root)
            leaf.incorporate_estimates(action_probs[0],rolloutReturn,rolloutReturn,up_to=self.root)

            
    def test_tree_search(self,mode="train"): 
        if mode == "train":      
            leaf = self.root.select_leaf()
        else:
            leaf = self.root.exploit_leaf()
            
        if leaf.is_done():
            value = self.TreeEnv.get_return(leaf.state)
            leaf.backup_value(value,value,up_to=self.root)
        else:
            graphState = self.getPytorchDataOfIntermediateAIGs([leaf])
            action_probs, _,aigEmbed = self.agent_netw.stepNgetEmbedding(graphState)
            print("\nAction probs (Intermediate nodes):")
            print(os.path.splitext(leaf.state)[0]+'.pt.zip')
            print(aigEmbed[0][:20])
            print(action_probs[0])
            ## Compute montecarlo return using the policy's best move rather than random policy
            startingAction = np.argmax(action_probs[0])
            value = self.TreeEnv.get_montecarlo_return(startingAction,leaf.state,leaf.depth)
            #leaf.incorporate_estimates(action_probs[0],values[0],value,up_to=self.root)
            leaf.incorporate_estimates(action_probs[0],value,value,up_to=self.root)
            
        factorValue = 1/(1-value)
        return factorValue
    
    
    def synergy_test_tree_search(self,mode="train"): 
        if mode == "train":      
            leaf = self.root.select_leaf()
        else:
            leaf = self.root.exploit_leaf()
            
        if leaf.is_done():
            value = self.TreeEnv.get_return(leaf.state)
            leaf.backup_value(value,value,up_to=self.root)
        else:
            graphState = self.getPytorchDataOfIntermediateAIGs([leaf])
            action_probs, _,aigEmbed = self.agent_netw.stepNgetEmbedding(graphState)
            action_probs[0] = np.power(action_probs[0],self.alphaValue)
            print("\nAction probs (Intermediate nodes):")
            print(os.path.splitext(leaf.state)[0]+'.pt.zip')
            print(aigEmbed[0][:20])
            print(action_probs[0])
            ## Compute montecarlo return using the policy's best move rather than random policy
            startingAction = np.argmax(action_probs[0])
            value = self.TreeEnv.get_montecarlo_return(startingAction,leaf.state,leaf.depth)
            #leaf.incorporate_estimates(action_probs[0],values[0],value,up_to=self.root)
            leaf.incorporate_estimates(action_probs[0],value,value,up_to=self.root)
            
        factorValue = 1/(1-value)
        return factorValue
    
    def embed_test_tree_search(self,mode="train"): 
        if mode == "train":      
            leaf = self.root.select_leaf()
        else:
            leaf = self.root.exploit_leaf()
            
        if leaf.is_done():
            value = self.TreeEnv.get_return(leaf.state)
            leaf.backup_value(value,value,up_to=self.root)
        else:
            graphState = self.getPytorchDataOfIntermediateAIGs([leaf])
            action_probs, _,fullEmbed = self.agent_netw.stepNgetFullEmbedding(graphState)
            action_probs[0] = np.power(action_probs[0],self.alphaValue)
            print("\nAction probs (Intermediate nodes):")
            print(os.path.splitext(leaf.state)[0]+'.pt.zip')
            #print(fullEmbed[0][:20])
            print(action_probs[0])
            ## Compute montecarlo return using the policy's best move rather than random policy
            startingAction = np.argmax(action_probs[0])
            value = self.TreeEnv.get_montecarlo_return(startingAction,leaf.state,leaf.depth)
            #leaf.incorporate_estimates(action_probs[0],values[0],value,up_to=self.root)
            leaf.incorporate_estimates(action_probs[0],value,value,up_to=self.root)
            
        factorValue = 1/(1-value)
        return factorValue,fullEmbed
        

    def pick_action(self):
        """
        Selects an action for the root state based on the visit counts.
        """
        if self.childType == 'max':
            action = np.argmax(self.root.child_Q) # MCTS max child
        else:
            action = np.argmax(self.root.child_N) # MCTS robust child
        return action

    def take_action(self, action):
        """
        Takes the specified action for the root state. The subsequent child
        state becomes the new root state of the tree.
        :param action: Action to take for the root state.
        """
        # Store data to be used as experience tuples.
        #ob = self.TreeEnv.get_obs_for_states([self.root.state])
        
        ob = self.root.ptstate
        self.obs.append(ob)
        self.searches_pi.append(self.root.visits_as_probs()) # TODO: Use self.root.position.n < self.temp_threshold as argument
        self.qs.append(self.root.Q)
        # Resulting state becomes new root of the tree.                                                                                                                           
        self.root = self.root.maybe_add_child(action)
        self.mc.append(self.root.averagedMonteCarlo)  # Add most promising child's averaged monte carlo rollout.
        del self.root.parent.children


def execute_episode(agent_netw,TreeEnv):
    """
    Executes a single episode of the task using Monte-Carlo tree search with
    the given agent network. It returns the experience tuples collected during
    the search.
    :param agent_netw: Network for predicting action probabilities and state
    value estimate.
    :param num_simulations: Number of simulations (traverses from root to leaf)
    per action.
    :param TreeEnv: Static environment that describes the environment dynamics.
    :return: The observations for each step of the episode, the policy outputs
    as output by the MCTS (not the pure neural network outputs), the individual
    rewards in each step, total return for this episode and the final state of
    this episode.
    """
    mcts = MCTS(agent_netw, TreeEnv,childType=CHILD_TYPE)
   
    mcts.initialize_search()

    # Must run this once at the start, so that noise injection actually affects
    # the first action of the episode.
    first_node = mcts.root.select_leaf()
    graphState = mcts.getPytorchDataOfIntermediateAIGs([first_node])
    probs, vals = agent_netw.step(graphState)
    ## Compute montecarlo return using the policy's first best move followed by random rather than entirely random policy
    startingAction = np.argmax(probs[0])
    first_node_rolloutReturn = TreeEnv.get_montecarlo_return(startingAction,first_node.state,first_node.depth) #resyn2 output will lead to DRAW or 0.
    first_node.incorporate_estimates(probs[0], vals[0], first_node_rolloutReturn,first_node)
    num_simulations_dict = [512,256,128,128,128,128,64,64,64,64]
    depth = 0
    while True:
        mcts.root.inject_noise() #NYU-MLDA Implemented in original version
        current_simulations = mcts.root.N

        # We want `num_simulations` simulations per action not counting
        # simulations from previous actions.
        while mcts.root.N < current_simulations + num_simulations_dict[depth]:
            mcts.tree_search()

        if num_simulations_dict[depth]>0:
            print("\nSimulation Depth: "+str(depth))
            print("\nSimulation iteration: "+str((mcts.root.N-current_simulations)))
            mcts.root.print_bfs_tree()
            print("_"*100)

        action = mcts.pick_action()
        mcts.take_action(action)
        depth+=1

        if mcts.root.is_done():
            break

    # Computes the returns at each step from the list of rewards obtained at
    # each step. The return is the sum of rewards obtained *after* the step.
    #terminalStateReturn = TreeEnv.get_return(mcts.root.state)
    #ret = [terminalStateReturn for _ in range(len(mcts.rewards))]
    #print("Terminal state return:"+str(terminalStateReturn))

    return (mcts.obs, mcts.searches_pi,mcts.mc,mcts.root.state)

def test_episode(agent_netw,maxTestBudget,csvFile,TreeEnv):
    """
    Executes a single episode of the task using Monte-Carlo tree search with
    the given agent network. It returns the experience tuples collected during
    the search.
    :param agent_netw: Network for predicting action probabilities and state
    value estimate.
    :param num_simulations: Number of simulations (traverses from root to leaf)
    per action.
    :param TreeEnv: Static environment that describes the environment dynamics.
    :return: The observations for each step of the episode, the policy outputs
    as output by the MCTS (not the pure neural network outputs), the individual
    rewards in each step, total return for this episode and the final state of
    this episode.
    """
    mcts = MCTS(agent_netw, TreeEnv,childType=CHILD_TYPE)
    mcts.initialize_search()
    first_node = mcts.root.select_leaf()
    graphState = mcts.getPytorchDataOfIntermediateAIGs([first_node])
    probs, vals,aigEmbed = agent_netw.stepNgetEmbedding(graphState)
    startingAction = np.argmax(probs[0])
    print("\n\n-- Testing episodes --\n\n")
    print(os.path.splitext(first_node.state)[0]+'.pt.zip')
    print(aigEmbed[0][:20])
    print("\nAction probs(Root node):")
    print(probs[0])
    first_node_rolloutReturn = TreeEnv.get_montecarlo_return(startingAction,first_node.state,first_node.depth) #resyn2 output will lead to DRAW or 0.
    first_node.incorporate_estimates(probs[0], vals[0], first_node_rolloutReturn,first_node)
    csvFileHandler = open(csvFile,'w')
    bestResultSoFar = 0
    testBudget=0
    while testBudget<maxTestBudget:
        currentMonteCarloValue = mcts.test_tree_search()
        if currentMonteCarloValue > bestResultSoFar:
            bestResultSoFar = currentMonteCarloValue
        csvFileHandler.write(str(currentMonteCarloValue)+","+str(bestResultSoFar)+"\n")
        testBudget+=1
        print(str(currentMonteCarloValue)+","+str(bestResultSoFar)+"\n")
    csvFileHandler.close()


def synergy_test_episode(agent_netw,maxTestBudget,alphaValue,csvFile,TreeEnv):
    """
    Executes a single episode of the task using Monte-Carlo tree search with
    the given agent network. It returns the experience tuples collected during
    the search.
    :param agent_netw: Network for predicting action probabilities and state
    value estimate.
    :param num_simulations: Number of simulations (traverses from root to leaf)
    per action.
    :param TreeEnv: Static environment that describes the environment dynamics.
    :return: The observations for each step of the episode, the policy outputs
    as output by the MCTS (not the pure neural network outputs), the individual
    rewards in each step, total return for this episode and the final state of
    this episode.
    """
    mcts = MCTS(agent_netw, TreeEnv,childType=CHILD_TYPE,alphaValue=alphaValue)
    mcts.initialize_search()
    first_node = mcts.root.select_leaf()
    graphState = mcts.getPytorchDataOfIntermediateAIGs([first_node])
    probs, vals,aigEmbed = agent_netw.stepNgetEmbedding(graphState)
    probs[0] = np.power(probs[0],alphaValue)
    startingAction = np.argmax(probs[0])
    print("\n\n-- Testing episodes --\n\n")
    print(os.path.splitext(first_node.state)[0]+'.pt.zip')
    print(aigEmbed[0][:20])
    print("\nAction probs(Root node):")
    print(probs[0])
    first_node_rolloutReturn = TreeEnv.get_montecarlo_return(startingAction,first_node.state,first_node.depth) #resyn2 output will lead to DRAW or 0.
    first_node.incorporate_estimates(probs[0], vals[0], first_node_rolloutReturn,first_node)
    csvFileHandler = open(csvFile,'w')
    bestResultSoFar = 0
    testBudget=0
    while testBudget<maxTestBudget:
        currentMonteCarloValue = mcts.synergy_test_tree_search()
        if currentMonteCarloValue > bestResultSoFar:
            bestResultSoFar = currentMonteCarloValue
        csvFileHandler.write(str(currentMonteCarloValue)+","+str(bestResultSoFar)+"\n")
        testBudget+=1
        print(str(currentMonteCarloValue)+","+str(bestResultSoFar)+"\n")
    csvFileHandler.close()

def embed_test_episode(agent_netw,maxTestBudget,csvFile,TreeEnv):
    """
    Executes a single episode of the task using Monte-Carlo tree search with
    the given agent network. It returns the experience tuples collected during
    the search.
    :param agent_netw: Network for predicting action probabilities and state
    value estimate.
    :param num_simulations: Number of simulations (traverses from root to leaf)
    per action.
    :param TreeEnv: Static environment that describes the environment dynamics.
    :return: The observations for each step of the episode, the policy outputs
    as output by the MCTS (not the pure neural network outputs), the individual
    rewards in each step, total return for this episode and the final state of
    this episode.
    """
    mcts = MCTS(agent_netw, TreeEnv,childType=CHILD_TYPE)
    mcts.initialize_search()
    first_node = mcts.root.select_leaf()
    graphState = mcts.getPytorchDataOfIntermediateAIGs([first_node])
    probs, vals,aigEmbed = agent_netw.stepNgetEmbedding(graphState)
    startingAction = np.argmax(probs[0])
    print("\n\n-- Testing episodes --\n\n")
    print(os.path.splitext(first_node.state)[0]+'.pt.zip')
    print(aigEmbed[0][:20])
    print("\nAction probs(Root node):")
    print(probs[0])
    first_node_rolloutReturn = TreeEnv.get_montecarlo_return(startingAction,first_node.state,first_node.depth) #resyn2 output will lead to DRAW or 0.
    first_node.incorporate_estimates(probs[0], vals[0], first_node_rolloutReturn,first_node)
    csvFileHandler = open(csvFile,'w')
    bestResultSoFar = 0
    testBudget=0
    embeddingList = []
    while testBudget<maxTestBudget:
        currentMonteCarloValue,curFullEmbedding = mcts.embed_test_tree_search()
        print("Current full embedding:")
        print(curFullEmbedding[0][:1])
        print(np.shape(curFullEmbedding[0]))
        if currentMonteCarloValue > bestResultSoFar:
            bestResultSoFar = currentMonteCarloValue
        csvFileHandler.write(str(currentMonteCarloValue)+","+str(bestResultSoFar)+"\n")
        testBudget+=1
        print(str(currentMonteCarloValue)+","+str(bestResultSoFar)+"\n")
        embeddingList.append(curFullEmbedding[0])
    csvFileHandler.close()
    return embeddingList
    
    


def get_init_graphEmbedding(agent_netw,TreeEnv):
    """
    Executes a single episode of the task using Monte-Carlo tree search with
    the given agent network. It returns the experience tuples collected during
    the search.
    :param agent_netw: Network for predicting action probabilities and state
    value estimate.
    :param num_simulations: Number of simulations (traverses from root to leaf)
    per action.
    :param TreeEnv: Static environment that describes the environment dynamics.
    :return: The observations for each step of the episode, the policy outputs
    as output by the MCTS (not the pure neural network outputs), the individual
    rewards in each step, total return for this episode and the final state of
    this episode.
    """
    mcts = MCTS(agent_netw, TreeEnv,childType=CHILD_TYPE)
    mcts.initialize_search()
    first_node = mcts.root.select_leaf()
    graphState = mcts.getPytorchDataOfIntermediateAIGs([first_node])
    _, _,aigEmbed = agent_netw.stepNgetEmbedding(graphState)
    return aigEmbed
