import time
from PVNet import PVNet2
import random
from collections import defaultdict, deque
from Game import Game, Player, Board
import numpy as np

# run through the training pipeline
class Train():
    def __init__(self, dimension=6, win_l=4):
        self.dimension = dimension
        self.win_l = win_l
        self.pv_net = PVNet2(dimension, dimension)
        self.data_collection = deque(maxlen=10000)
        self.n_games = 300
        self.batch_size = 512
        self.epochs = 5
        self.lr_multiplier = 1.0
        self.kl_targ = 0.02
        self.data_len = []
        self.game_len = 0
        
    # collect data through self play
    def collect_data_self_play(self, show=False, n_rounds=1):
        # perform n rounds of self play
        for i in range(n_rounds):
            #print("collect", self.pv_net)
            player = Player(self.pv_net)
            game = Game(player, self.dimension, self.win_l, show)
            data = list(game.start())[:]
            #print(len(data))
            #self.data_len += len(data)
            #print(data)
            self.game_len = len(data)
            data = self.get_equi_data(data)
            self.data_len.append(len(data))
            #print("datalen", len(data))
            self.data_collection.extend(data)
      
    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.dimension, self.dimension)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data
    # run the training process
    def run(self, n_games= 300, display_games=50, l_rate_m=1.0):
        #self.lr_multiplier /= l_rate_m
        self.n_games = n_games
        for i in range(self.n_games):
            s = time.time() 
            self.collect_data_self_play()
            if ((i + 1) % display_games == 0):
                player = Player(self.pv_net, n_simulations=800)
                board = Board(self.dimension, self.win_l)
                while True:
                    move, p = player.get_move(board, self_play=False)
                    board.make_move(move)
                    pred = self.pv_net.predict(board.get_state().reshape(-1, 4, 6, 6))
                    print("prediction:\n", pred[0], "\n", pred[1])
                    print("p", p.reshape(-1, self.dimension))
                    print("move", move)
                    print("board\n", board.board)
                    if board.has_ended():
                        print(-board.player, "won")
                        break
            
            if ((i + 1) % 100 == 0):
                self.lr_multiplier /= 1.5
            #print(time.time() - s)
            #if (len(self.data_collection)) > self.batch_size:
            print("game", i, "completed in", time.time() - s, "s", self.game_len, "steps")
            if (i >= 300):
                l = self.data_len.pop(0)
                print("length", l, "data l", len(self.data_collection))
                for i in range(l):
                    self.data_collection.popleft()
                    
                print("data l after", len(self.data_collection))
            #print(self.data_collection[0])
            #print(states[0].shape)
            #print(probs[0].reshape(-1, self.dimension**2).shape)
            #print(winners[0].reshape(1, 1).shape, winners)
            #print(len(self.data_collection))
            if len(self.data_collection) > self.batch_size:
                s = time.time()
                # train neural network
                
                batch = random.sample(self.data_collection, self.batch_size)
                #s = np.array([d[0] for d in batch])
                #print("s1 shape", s.shape)
                states = np.array([d[0] for d in batch])
                #print("s shape", states.shape)
                probs = np.array([d[1] for d in batch])
                #print("p shape", probs.shape)
                winners = np.array([d[2] for d in batch]).reshape(-1, 1)
                #print("w shape", winners.shape)
                old_probs, old_v = self.pv_net.predict(states, v=False)
                #print(np.array(old_probs).shape, np.array(old_v).shape)
                for j in range(self.epochs):
                    #print(batch.shape)
                    
                    #print(states.shape, probs.shape, winners.shape)
                    loss, entropy = self.pv_net.train(states, probs, winners, 2e-3 * self.lr_multiplier)
                    
                    new_probs, new_v = self.pv_net.predict(states, v=False)
                    print("training", j, "lr_mult", self.lr_multiplier, "loss", loss, "entropy", entropy)
                    
                    kl = np.mean(np.sum(old_probs * (
                        np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                        axis=1))
                    
                    if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                        break
                # adaptively adjust the learning rate
                print("kl", kl)
                
                if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
                    self.lr_multiplier /= 1.5
                elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
                    self.lr_multiplier *= 1.5
                
                
                    
                print("completed in", time.time() - s, "s")
    def save(self, path):
        self.pv_net.save(path)

model = Train()
model.run()