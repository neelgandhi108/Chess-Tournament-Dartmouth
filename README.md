# Chess-Tournament-Dartmouth

This project was develop for chess tournment held at Dartmouth College.
Algorithm was developed using Minmax,Alpha-beta pruning,Quiescence search and stockfish.Training took about 10-12 hours of training to reduce loss function.The project was developed out of curiosity and if you find it help please star the repository.

## Installation
## Libraries
There are multiple data visualiation,machine learning and chess librarires that could be installed using requirements.txt.

    pip install requirements.txt



### StockFish 
It difficult to install stockfish in windows or mac.For more guidance,follow the link - [Download Stockfish 15 - Stockfish - Open Source Chess Engine (stockfishchess.org)](https://stockfishchess.org/download/)

OR 

Directly use my jupyter notebooks for testing your algorithm against RL algorithm - **Chess_Tournament_Dartmouth_Testing.ipynb**
For generating training data and pickle file - **Chess_Tournament_Dartmouth_Training.ipynb**


## Chess AI

Chess has long been used to test the capabilities of artificial intelligence. The significance of this defeat stems from the fact that chess is a notoriously difficult game for a computer to play, especially given that traditional tabular computation methods to defeat the game are nearly impossible due to the game's 10^40^  possible positions and 10^120^ possible game outcomes.

As a result, we wanted to experiment with reinforcement learning in order to build a chess AI that is roughly symbolic of an average player, capable of displaying some good traits while playing but ultimately capable of easily defeating a weak opponent solely through reinforcement learning.

The final method we considered was model-based reinforcement learning. In this technique, we simulate the board and traverse and collect board performance samples to aid in training prior to execution. This was ultimately the approach we chose because it is easily compatible with a variety of conventional techniques used to improve Chess engine performance, and the model for chess is already provided by the python-chess library. As a result, we chose to use this approach in conjunction with a deep learning model built with Keras. This technique necessitates both a method for evaluating the state of chess boards in order to assign appropriate rewards while training the neural network.Second, in order for optimal learning to occur, a method of moving through game states while simulating chess will be required

## MinMax 

Minimax is one of the algorithms that is thought to be used to traverse all possible game states. In chess, we consider the evaluation of the board in terms of all possible moves, and then we consider our opponents' best responses and choose the move with the worst outcome for our opponent. As a result, our advantage over them is maximized. This algorithm is notoriously inefficient because the number of possible moves grows exponentially as depth increases, becoming unmanageable after a few layers.

## Alpha-Beta pruning
Minimax's alpha-beta pruning is a minor tweak that dramatically improves performance. The algorithm monitors two variables, alpha and beta, which track the scores of both the player we want to maximize and the player we want to minimize. This technique allows large sections of the tree to be avoided without missing any better moves. Furthermore, if a move appears to be good, it is investigated further, and if it results in a large material loss, it can be avoided.

## Quiescence search
This is the method of search that we ultimately chose.
With the addition of a Quiescence search to avoid the horizon effect, which occurs when a depth search ends and potentially ignores immediate ramifications a level below the search's end point. A move that does not include a Quiescence search may take material but inadvertently offer check to the opponent the following turn. As a result, Quiescence search digs deeper to ensure that the search's terminal nodes are as good as they appear before returning the best state.

## Evaluation Method

Another method of evaluation is to use an existing world-class engine.
Stockfish is the current best engine that is easily pluggable into our chess library. Based on the complicated heuristics used to power the game engine, this engine can easily value the game balance in relation to a plethora of factors. This provides a much more precise reading of the score and, if perfectly learned by our network, would allow for performance comparable to a very competent engine.


## Neural Network
The design of the network was the next concern after the methods of training the network were planned. For the network, there is one existing piece of
work was used that used chess board inputs to predict quality as a scalar, but this prediction was simply a guesstimate of which side was likely to win - ours is a qualification of how good a board is at any given point for our side. As a result, we modeled our network similarly by stacking LSTM layers with dropout. Experimentation revealed that an additional dense layer improved performance, and that the activation function was best suited with a scaled exponential linear unit function.This is most likely due to the fact that the value of an imminent checkmate is so much higher than that of a typical strong board position that this function provides a very accurate map with training.

## Testing

In terms of performance, it is not uncommon to obtain up to 10k samples from which to learn from a single game due to the number of paths produced during the alpha-beta pruned minimax. As a result, the datasets to learn from for each game are massive, but the loss appears to be reduced very nicely and becomes asymptotic after about 64 epochs.

As a result, the overall game is played fairly well. When using our Alpha-Beta algorithm in conjunction with this accurate neural network predictor, future move states are reasonably well predicted, and as a result, a number of good chess qualities emerge.

As seen in the image, a large portion of the material is covered by another piece; there are multiple threats at once, and pieces that should be developed last are being developed last. This means no unnecessary movement of the king, which is a concerning flaw that occurs frequently in early learning. All of this was accomplished in ten games by training on a set of data we created after each game that included every potential move the engine could have made as well as the score stockfish had associated with that boardstate.

## Shortcomings

Fundamentally, this engine does not play the best openings and has poor endgame performance. Openings are well known in chess, so it's easy to tell what is and isn't good practice - from experience, the way the engine plays results in early blunders that certainly weaken it. Furthermore, there is the late game. The engine's performance is also subpar; I suspect that under more rigorous testing and conditions, it will stalemate.
Another issue is that because the engine does not know the value of some pieces, it occasionally plays them too aggressively for too little reward. This includes putting the queen in perilous situations with little to ensure that she does not capitulate.
