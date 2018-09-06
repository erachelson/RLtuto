### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, sgd
import numpy as np
import random
from IPython.display import clear_output

# The Deep Q network

model = Sequential()

model.add(Dense(150, init='lecun_uniform', input_shape=(9,)))
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(Dense(150, init='lecun_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(9, init='lecun_uniform'))
model.add(Activation('linear'))
model.compile(loss='mse', optimizer="rmsprop")

print(model.summary())

# Parameters for Q-learning

epochs = 1000
gamma = 0.99 # discount factor
epsilon = 1 # epsilon-greddy

update=0 
alpha=0.1 # learning rate
experience_replay=True
batchSize = 40 # mini batch size
buffer = 1000
replay = [] # init vector buffer
h=0 # current size of the vector buffer

# Initialize game

jeu = oxo()

# Learn

for i in range(epochs):
    jeu = oxo()
    state = jeu.board
    current_player = jeu.player(state)
    
    while(jeu.end_game==0):
        current_player = jeu.player(state)
        qval = model.predict(np.array(state).reshape(1,len(state)), batch_size=batchSize)
        if (random.random() < epsilon): # exploration exploitation strategy    
            rand = np.random.randint(0,len(jeu.actions))
            action = jeu.actions[rand]
        else: #choose best action from Q(s,a) values
            qval_av_action = [-9999]*9
            for ac in jeu.actions:
                qval_av_action[ac] = qval[0][ac]
            action = (np.argmax(qval_av_action))
        #Take action, observe new state S'
        #Observe reward
        jeu.play(state, action)
        new_state = jeu.board
        # choose new reward values
        reward = -5
        if jeu.payoff(current_player, new_state) == current_player:
            reward = 2
        if jeu.payoff(current_player, new_state) == 0:
            reward = 1
        if jeu.payoff(current_player, new_state) == -1:
            reward = 0
            
        if not experience_replay:
            #Get max_Q(S',a)
            newQ = model.predict(np.array(new_state).reshape(1,len(state)), batch_size=batchSize)
            maxQ = np.max(newQ)
            y = np.zeros((1,9))
            y[:] = qval[:]
            if reward != 0: #non-terminal state
                update = (reward + gamma * maxQ)
            else:
                update = reward
            y[0][action] = update #target output
            print("Game #: %s" % (i,))
            model.fit(np.array(state).reshape(1, len(state)), y, batch_size=batchSize, nb_epoch=1, verbose=1)
            state = new_state
            clear_output(wait=True)
            
        else:
            #Experience replay storage
            if (len(replay) < buffer): #if buffer not filled, add to it
                replay.append((state, action, reward, new_state))
            else: #if buffer full, overwrite old values
                if (h < (buffer-1)):
                    h += 1
                else:
                    h = 0
                replay[h] = (state, action, reward, new_state)
                #randomly sample our experience replay memory
            
            if(len(replay)>batchSize):
                minibatch = random.sample(replay, batchSize)
                X_train = []
                y_train = []
                for memory in minibatch:
                    #Get max_Q(S',a)
                    old_state, action, reward, new_state = memory
                    old_qval = model.predict(np.array(old_state).reshape(1,len(old_state)), batch_size=1)
                    newQ = model.predict(np.array(new_state).reshape(1,len(new_state)), batch_size=1)
                    maxQ = np.max(newQ)
                    y = old_qval[:]
                    if reward != 0: #non-terminal state
                        update = (reward + (gamma * maxQ))
                    else: #terminal state
                        update = reward
                    y[0][action] = update
                    X_train.append(np.array(old_state).reshape(len(old_state),))
                    y_train.append(np.array(y).reshape(9,))
    
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                print("Game #: %s" % (i,))
                model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=1)
                state = new_state
            clear_output(wait=True)
        # update exploitation / exploration strategy
        if epsilon > 0.1:
            epsilon -= (1.0/epochs)
    
        # save the model every 1000 epochs
        if i%1000 == 0:
            model.save("model_dql_oxo_dense.dqf")


