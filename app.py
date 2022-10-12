from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.datasets import cifar10
from Game import Game, Player, Board
from PVNet import PVNet2

# pip install -U flask-cors

# initialize our Flask application
app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'



dimension = 6
win_l = 4
pv_net = PVNet2(dimension, dimension)
pv_net.restore('n2-900.model')
player = Player(pv_net, n_simulations=1600)
board = Board(dimension, win_l)

@app.route("/")
@cross_origin()
def helloWorld():
    return "Hello, cross-origin-world!"


@app.route("/initialize", methods=["GET"])
def init():
    if request.method == 'GET':
        print("init")
        board.reset()
        return jsonify("init")

@app.route("/ai", methods=["GET"])
def predict_move():

    i = 0
    if request.method == 'GET':

        #board = np.array(request.args.get('board'))
        i = int(request.args.get('idx'))
        #print(board)
        print('i:', i)

        return jsonify(id = 1, test = "t")

@app.route("/make_move", methods=["GET"])
def make_move():
    if request.method == 'GET':
        #board = np.array(request.args.get('board'))
        i = int(request.args.get('i'))
        j = int(request.args.get('j'))
        print("i", i)
        board.make_move(i * dimension + j)
        print("ava", board.availables)
        print(board.board)
        print("result", board.has_ended())
        result = False
        if (board.has_ended()):
            result = True
        pred = pv_net.predict(board.get_state().reshape(-1, 4, 6, 6))

        return jsonify(result=result, win_prob=str(round(pred[1], 4)))


@app.route("/get_move", methods=["GET"])
def get_move():
    if request.method == 'GET':
        print("availables", board.availables)
        #move = 0
        pred = [1,2]
        
        move, p = player.get_move(board, self_play=False)
        board.make_move(move)
        pred = pv_net.predict(board.get_state().reshape(-1, 4, 6, 6))
        print("prediction:\n", pred[0], "\n", pred[1])
        print("p", p.reshape(-1, dimension))
        print("move", move)
        print("board\n", board.board)
        result = False
        if board.has_won():
            print(-board.player, "won")
            result = True
        
        print(round(pred[1], 2))
        return jsonify(pos=move, result=result, win_prob=str(round(pred[1], 4)))

@app.route("/undo", methods=["GET"])
def undo():
    if request.method == "GET":
        board.undo()
        print(board.board)
        return jsonify(result=True)

#  main thread of execution to start the server
if __name__ == '__main__':
    app.run(debug=True)



