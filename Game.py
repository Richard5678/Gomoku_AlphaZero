import random
import numpy as np
from collections import defaultdict, deque

class Node():
    def __init__(self, parent, p):
        self.parent = parent
        self.children = {}
        self.Q = 0
        self.c = 5
        self.visits = 0
        self.P = p
    
    # get the value Q + U
    def get_value(self):
        return self.Q + self.c * self.P * np.sqrt(self.parent.visits) / (1 + self.visits)
    
    # expand the leaf node 
    def expand(self, actions, probs):
        #print(actions, probs)
        for i in range(len(actions)):
            if actions[i] not in self.children:
                self.children[actions[i]] = Node(self, probs[i])
          
    # back propagate the value from leaf to root
    def update(self, value):
        self.visits += 1
        self.Q += 1.0 * (value - self.Q) / self.visits
        if self.parent:
            self.parent.update(-value)


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs
            
import copy, time
# Monte Carlo Tree Search
class MCTS():
    def __init__(self, pv_net, c, n_simulations):
        self.root = Node(None, 1.0)
     
        self.pv_net = pv_net
        self.c = c
        self.n_simulations = n_simulations
    
    # run n simulations of mcts
    def run(self, board, temp=1):
        #visits = [self.root.children[c].visits for c in self.root.children]
        #print("visits b", visits)
        for i in range(self.n_simulations):
            #if i % 100 == 0:
               # print(i, " simulations")
            self.traverse(self.root, copy.deepcopy(board))
            
        #print("time", time.time() - start)
        visits = [self.root.children[c].visits for c in self.root.children]
        actions = [a for a in self.root.children]
        #print("actions", actions)
        #print("visits", visits, len(visits))
        actions_p = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        #print("visits", visits, np.array(visits).shape)
        #print("actions", actions_p)
        return actions, actions_p
    
    # perform one simulation fo mcts
    def traverse2(self, root, board, start):
        ptr = self.root
        while True:
            if len(ptr.children) == 0: break
            action, ptr = max(ptr.children.items(), key=lambda n: n[1].get_value())
            board.make_move(action)
        # base case, at leaf node
        #print("t before", time.time() - start)
        probs, value = self.pv_net.predict(board.get_state().reshape(-1, 4, 6, 6))
        #print("t after", time.time() - start)   
            
        if board.has_ended():
            #print("2")
            # use the true reward if game has ended
            if len(board.availables) == 0: value = 0
            else: value = board.player
        else:
            #print("3")
            #print("probs", probs.shape)
            ptr.expand(board.availables, probs[board.availables]) # expand the tree

        # backpropagate the value from leaf to root
        #print("value", value)
        ptr.update(-value)
    
         
    # perform one simulation fo mcts
    def traverse(self, root, board):
        # base case, at leaf node
        if len(root.children) == 0:
            #print("1")
            start = time.time()
            probs, value = self.pv_net.predict(board.get_state().reshape(-1, 4, 6, 6))
            #print(value)
            #print("t", time.time() - start)
            #print("p", probs)
            if board.has_ended():
                #print("2")
                #print(board.board)
                # use the true reward if game has ended
                if len(board.availables) == 0: value = 0
                else: value = -1
            else:
                #print("3")
                #print("probs", probs.shape)
                root.expand(board.availables, probs[board.availables]) # expand the tree
                
            # backpropagate the value from leaf to root
            #print("value", value)
            root.update(-value)
            return
            
        # Greedily select the children node with the max(Q + U) until leaf 
        #      where Q is the average of all values of nodes in the subtree and 
        #      U = c * p * sqrt(parent.visits) / (1 + visits)
        action, child = max(root.children.items(), key=lambda n: n[1].get_value())
        board.make_move(action)
        self.traverse(child, board)
    
        
    def forward(self, move):
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            self.root = Node(None, 1.0)
        
        
class Player():
    def __init__(self, pv_net, n_simulations=400):
        #print("p", pv_net)
        self.mcts = MCTS(pv_net, 1, n_simulations)
        self.n_simulations = n_simulations
        self.pv_net = pv_net
        
    def get_move(self, board, self_play=True):
        actions, actions_p = self.mcts.run(board)
        
        move = np.random.choice(
                    actions,
                    p=0.75*actions_p + 0.25*np.random.dirichlet(0.3*np.ones(len(actions_p)))
                )
                # update the root node and reuse the search tree
            
        if not self_play:
            move = actions[np.argmax(actions_p)]
            
        move_p = np.zeros(board.dimension**2)
        move_p[list(actions)] = actions_p
        if self_play:
            self.mcts.forward(move)
        else:
            self.mcts.forward(-1)
        return move, move_p
        


from itertools import groupby
class Board():
    '''Board used in the Gomoku'''
    def __init__(self, dimension, win_l):
        self.dimension = dimension
        self.board = np.zeros((dimension, dimension))
        self.win_l = win_l         
        self.prev_move = []
        self.player = 1
        self.availables = list(range(dimension * dimension))
        
    def reset(self):
        self.board = np.zeros((self.dimension, self.dimension))
        self.prev_move.clear()
        self.player = 1
        self.availables = list(range(self.dimension * self.dimension))
    
    def undo(self):
        self.availables.append(self.prev_move[-1])
        self.availables.append(self.prev_move[-2])
        self.board[self.prev_move[-1] // self.dimension][self.prev_move[-1] % self.dimension] = 0
        self.board[self.prev_move[-2] // self.dimension][self.prev_move[-2] % self.dimension] = 0
        self.prev_move = self.prev_move[:-2]
        print(self.player)

    def make_move(self, pos):
        self.board[pos // self.dimension][pos % self.dimension] = self.player
        self.prev_move.append(pos)
        self.player = -self.player
        self.availables.remove(pos)
        
    def has_won(self):
        def check(a):
            length = max(sum(1 for i in g) for k, g in groupby(a))
            #print("length", length)
            return length >= 4 and a[a.shape[0] // 2] != 0
            
            return False
        
        i = self.prev_move[-1] // self.dimension
        j = self.prev_move[-1] % self.dimension
        
        # horizontal
        #print("i", i, "j", j)
        #print("l", self.board[i, :])
        if check(self.board[i, :]): return True
        # vertical
        #print("2", self.board[:, j])
        if check(self.board[:, j]): return True
        # diagonal \
        #print("3", self.board.diagonal(j - i))
        if check(self.board.diagonal(j - i)): return True
        # diagonal /
        #print("4", np.fliplr(self.board).diagonal(self.dimension - j - i - 1))
        return check(np.fliplr(self.board).diagonal(self.dimension - j - i - 1))
    
    def has_ended(self):
        return len(self.availables) != self.dimension**2 and (self.has_won() or (len(self.availables) == 0))
    
    def get_state(self):
        #print(self.board.shape)
        state = np.zeros((4, self.board.shape[0], self.board.shape[1]))
        state[0] = (self.board == self.player).astype(int)
        state[1] = (self.board == -self.player).astype(int)
        last = np.zeros(self.board.shape)
        if self.prev_move:
            last[self.prev_move[-1] // self.dimension][self.prev_move[-1] % self.dimension] = 1
        state[2] = last
        if self.player == 1:
            state[3] = np.ones(self.board.shape)
        else:
            state[3] = np.zeros(self.board.shape)
        
        return state
        


class Game():
    '''A Single Game of Gomoku'''
    def __init__(self, player, dimension, win_l, show):
        self.board = Board(dimension, win_l)
        self.states = []
        self.probs = []
        self.players = []
        self.opponent = {}
        self.player = player
        self.show = show
        
    # start game
    def start(self):
        i = 0
        #s = time.time()
        while True:
            move, p = self.player.get_move(self.board)
            i += 1
            #print("move", i, "time", time.time() - s)
            
            # before each move, save the the board state, probabilities of next actions and the current player at the turn
            #print("state after one move:\n", self.board.get_state())
            self.states.append(self.board.get_state())
            #print("current states", self.board.get_state())
            self.probs.append(p)
            #print("probs", p)
            self.players.append(self.board.player)
            #print("current player", self.board.player)
            
            self.board.make_move(move)
            if self.show:
                print(self.board.board)
            
            if self.board.has_ended():
                #print("1", self.board.board)
                winners = np.zeros(len(self.players))
                # someone wins
                if len(self.board.availables) > 0:
                    winner = -self.board.player
                    winners[np.array(self.players) == winner] = 1.0
                    winners[np.array(self.players) != winner] = -1.0
                '''
                for i in winners:
                    print("winner", i)
                for i in self.states:
                    print("s", i)
                for i in self.probs:
                    print("p", i)
                '''
                #print("winners", winners)
                return zip(self.states, self.probs, winners)
            
        
    def ended(self):
        return self.board.has_ended()