
import tensorflow as tf
import chess
import chess.svg
import numpy as np
import chess.pgn
import time
import asyncio
import chess.engine
import datetime
import keras
import random
import matplotlib.pyplot as plt
from chess.engine import Cp, Mate, MateGiven
from IPython.display import SVG
from IPython.display import clear_output
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Dropout
from keras.layers import LSTM
from numpy import array
import pickle as pkl

class ChessNeelAI:
    def __init__(self,engine,model) -> None:
        # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        self.engine = engine
        self.model = model

    def processScore(self,estScore):
        estScore = str(estScore)
        if ('#' in estScore):
            estScore = self.estScore.replace('#','')
            estScore = int(estScore)
            if(estScore > 0):
                estScore = int((10-estScore)*1000)
            else:
                estScore = int((-10-estScore)*1000)
        print(estScore)
        estScore = int(estScore)
        return estScore

    def getScore(self):
        info = self.engine.analyse(board, chess.engine.Limit(time=0.1))
        if board.turn: #white
            score = -(int(self.processScore(info.get("score"))))
        else:
            score = (int(self.processScore(info.get("score"))))
        if (score > 1000):
            score = score / 3
        return score
        
        
    def convertBoard(self):
        l = np.zeros(64)
        for sq in chess.scan_reversed(board.occupied_co[chess.WHITE]):
            l[sq] = board.piece_type_at(sq)
        for sq in chess.scan_reversed(board.occupied_co[chess.BLACK]):
            l[sq] = -board.piece_type_at(sq)
        l = l.reshape((8,8))
        l = np.flip(l,0)
        return l

    def initTrainSet(self):
        global X_train
        global Y_train
        X_train= np.array([], dtype=np.float64)
        Y_train= np.array([], dtype=np.float64)
        
    def addTrainSet(self):
        global X_train
        global Y_train
        tmp = self.convertBoard()
        tmp = array(tmp)
        tmp = tmp.reshape((1,tmp.shape[0],tmp.shape[1]))
        X_train = np.append(X_train,tmp)
        Y_train = np.append(Y_train,array(self.getScore()))
        
    def resizeDataSet(self):
        global X_train
        global Y_train
        X_train = X_train.reshape((Y_train.shape[0],8,8))

    def deepEval(self):
        x= np.array([], dtype=np.float64)
        tmp = self.convertBoard()
        tmp = array(tmp)
        tmp = tmp.reshape((1,tmp.shape[0],tmp.shape[1]))
        return (self.model.predict(tmp))

    def attemptTraining(self):
        global X_train
        global Y_train
        tf.keras.utils.normalize(
        Y_train, axis=0, order=2)
        print('normalising')
        print(Y_train)
        history = self.model.fit(X_train, Y_train,
                        batch_size=512,
                        epochs=3,shuffle=True)
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def quiesce2(self, alpha, beta ):
        stand_pat = self.deepEval()
        if( stand_pat >= beta ):
            return beta
        if( alpha < stand_pat ):
            alpha = stand_pat
        for move in board.legal_moves:
            if board.is_capture(move):
                board.push(move)
                score = -self.quiesce2( -beta, -alpha )
                board.pop()
                if( score >= beta ):
                    return beta
                if( score > alpha ):
                    alpha = score  
        return alpha

    def alphabeta(self, alpha, beta, depthleft ):
        bestscore = -9999
        if( depthleft == 0 ):
            return self.quiesce2( alpha, beta )
        for move in board.legal_moves:
            board.push(move)
            score = -self.alphabeta2( -beta, -alpha, depthleft - 1 )
            board.pop()
            if( score >= beta ):
                return score
            if( score > bestscore ):
                bestscore = score
            if( score > alpha ):
                alpha = score   
        return bestscore

    def selectmove2(self):
        bestMove = chess.Move.null()
        bestValue = -99999
        alpha = -100000
        beta = 100000
        for move in board.legal_moves:
            board.push(move)
            boardValue = -self.alphabeta2(-beta, -alpha, 1)
            if boardValue > bestValue:
                bestValue = boardValue;
                bestMove = move
            if( boardValue > alpha ):
                alpha = boardValue
            board.pop()
        return bestMove

    def quiesce(self, alpha, beta ):
        stand_pat = self.deepEval()
        if( stand_pat >= beta ):
            return beta
        if( alpha < stand_pat ):
            alpha = stand_pat
        for move in board.legal_moves:
            if board.is_capture(move):
                board.push(move)
                score = -self.quiesce( -beta, -alpha )
                self.addTrainSet()
                board.pop()
                if( score >= beta ):
                    return beta
                if( score > alpha ):
                    alpha = score  
        return alpha

    def alphabeta2(self, alpha, beta, depthleft ):
        bestscore = -9999
        if( depthleft == 0 ):
            return self.quiesce( alpha, beta )
        for move in board.legal_moves:
            board.push(move)
            score = -self.alphabeta( -beta, -alpha, depthleft - 1 )
            self.addTrainSet()
            board.pop()
            if( score >= beta ):
                return score
            if( score > bestscore ):
                bestscore = score
            if( score > alpha ):
                alpha = score   
        return bestscore

    def selectmove(self):
        bestMove = chess.Move.null()
        bestValue = -99999
        alpha = -100000
        beta = 100000
        for move in board.legal_moves:
            board.push(move)
            self.addTrainSet()
            boardValue = -self.alphabeta(-beta, -alpha, 1)
            if boardValue > bestValue:
                bestValue = boardValue;
                bestMove = move
            if( boardValue > alpha ):
                alpha = boardValue
            board.pop()
        return bestMove

    def randomMove(self):
        randomMove = []
        for move in board.legal_moves:
            randomMove.append(move)
        randVal = random.randint(0,(len(randomMove)-1))
        print(randomMove[randVal])
        board.push(randomMove[randVal])
        
    def choose_move(self,time):
        mov = self.selectmove2()
        print(mov)
        board.push(mov)


if __name__ == '__main__':
    
    #to load it
    with open("train.pkl", "rb") as f:
        X_train, Y_train = pkl.load(f)

    board = chess.Board()
    model = Sequential()
    X_size = np.zeros((1, 8, 8), dtype=int)
    model.add(LSTM(1024, input_shape=X_size.shape[1:], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(1024, return_sequences=True))
    model.add(Dropout(0.2))
    # model.add(LSTM(1024,return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(1024,return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(1024, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(Dense(1, activation='selu'))

    model.compile(loss="mse", optimizer="rmsprop")
    engine = chess.engine.SimpleEngine.popen_uci("stockfish_14_x64.exe")
    chessAI = ChessNeelAI(engine,model)
    chessAI.initTrainSet()
    while not board.is_game_over():
        if board.turn: #white
            chessAI.randomMove()
        else: #Black
            chessAI.choose_move(time)
        clear_output()
    chessAI.resizeDataSet()

