# -*- coding: utf-8 -*-

"""
Created on Thu Oct 27 23:29:47 2022

@author: thesc
"""

from copy import deepcopy
import numpy as np
from random import choice
import tensorflow as tf
from tensorflow import keras
from Threes_game import *
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
import os
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import sys
args = sys.argv
import tracemalloc
tracemalloc.start()

def disp_training_dict():
    for iter_num in training_dict:
        try:
            iter_num +1
        except:
            print(str(iter_num) + ': '+ str(training_dict[iter_num]))
            continue
        for model in training_dict[iter_num].keys():
            if model =='max_score':
                print('max_score: ' + str(training_dict[iter_num][model]['max_score']))
                continue
            print('Dense_Layers: ' + str(training_dict[iter_num][model]['Dense_Layers']))
            print('\n'.join(['layers:'] + [str(layer) for layer in training_dict[iter_num][model]['model'].layers]))
            print('Score: ' + str(training_dict[iter_num][model]['score']))

'''
def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
    return output
'''

def generate_model_dict(load_model_path = None):
    model_dict = {i:{'Dense_Layers':i} for i in range(numModels)}
    if load_model_path:
        for i in model_dict.keys():
            #clone
            model1 = tf.keras.models.load_model(load_model_path) # Expected Path: 'best_models/best_model_'+str(iter_num)+'_'+str(max_score)
            if i == 0:
                model_dict[i]['model'] = model1
            #mutate
            else:
                for layerNum, layer in enumerate(model1.layers):
                    if 'flatten' in layer.name:
                        continue
                    weights, biases = layer.get_weights()
                    weights1 = copy(weights)
                    weights1[choice(range(len(weights)))] = 0
                    model1.layers[layerNum].set_weights([weights1,biases])
            model_dict[i]['model'] = model1
            model_dict[i]['score'] = 0
    else:
        for i in model_dict.keys():
            layer_list = [Flatten(input_shape = (5,4)), Dense(units = 64,activation = 'relu')] + [Dense(units=64, activation='relu') for j in range(model_dict[i]['Dense_Layers'])] + [Dense(units=4, activation='softmax')]
            #layer_list = [Input(shape=(17,)),Dense(17,activation = 'relu')] + [Dense(units=64, activation='relu') for j in range(model_dict[i]['Dense_Layers'])] + [Dense(units=4, activation='softmax')]
            model_dict[i]['model'] = keras.Sequential(layer_list)
            model_dict[i]['score'] = 0
            #models.append(model)
    return model_dict

def get_best_model_path():
    saved_model_paths = os.listdir(best_models_folder)
    max_score = 0
    for model_path in saved_model_paths:
        a,b,iter_num,score = model_path.split('_')
        if int(score) > max_score:
            max_score = int(score)
            best_model_path = model_path            #
    return best_models_folder +'/'+ best_model_path


best_models_folder = 'best_models'
def isSavedModel(path = best_models_folder):
    return any(os.listdir(path))




directions = ['left','right', 'up', 'down']
training_dict = {'numIterations': 100}
numModels = 10
#models = []
# Build initial models
grid1 = Grid()
training_mode = 'new' # old or new
evolution_mode = 'train' # train or evolve
if isSavedModel():
    model_dict = generate_model_dict(load_model_path = get_best_model_path())
else:
    model_dict = generate_model_dict()



# Train a model that won't pick a direction that doesn't work
        #Could I train it to guess what will be available on the next move?

def isThreshMet():
    return all(thresh_bools)

def make_thresh_bools(predictions, training_thresh,movable_directions):
    thresh_bools = [False for i in range(len(directions))]
    for i in range(len(directions)):
        if directions[i] in movable_directions:
            if predictions[0][i] > training_thresh:
                thresh_bools[i] = True
            else:
                thresh_bools[i] = False 
        else: #not directions[i] in movable_directions
            if predictions[0][i] < training_thresh:
                thresh_bools[i] = True
            else:
                thresh_bools[i] = False
    return thresh_bools
'''#starter model
layer_list = [Flatten(input_shape = (5,4)), Dense(units = 64,activation = 'relu')] + [Dense(units=64, activation='relu') for j in range(9)] + [Dense(units=4, activation='softmax')]
mymodel1 = keras.Sequential(layer_list) 
''' 

def Train_No_Bad_Direction():
#    pass
#if True:
    mymodel1 = tf.keras.models.load_model('Model_makes_possible_moves_only')
    numGames = 1 # IF I make it more than 20, it tends to blow up my pc...
    model = mymodel1
    train_counts = []
    for i in range(numGames):
        train_count = 0
        grid1 = Grid()
        while grid1.isMovable():
            predictions = model(np.array([grid1.tensor]))
            predictions1 = predictions.numpy()
            max1 = predictions1.max()
            ind1 = np.where(predictions1[0] == max1)[0][0]
            print('Predictions: ' + str(predictions1))
            print('Directions : [[___left___ __right__ ____up____ ___down___]]') 
            print('Chosen Direction: ' +str(directions[ind1]))
            #print('train_counts: ' + str(sum(train_counts)))
            print('game num: ' + str(i) + ' out of '+ str(numGames))
            if grid1.isMovable(directions[ind1]):
                grid1.update_grid(directions[ind1])
                #record the score
                score = grid1.calculate_score()
            else:
                movable_directions = [direction for direction in directions if grid1.isMovable(direction)]
                training_threshold = 1/len(movable_directions)*.75
                #movability = [1 for direction in directions if grid1.isMovable(directions) else 0]
                thresh_bools = make_thresh_bools(predictions1, training_threshold, movable_directions)
                #movable_directionsbreakME
                while not all(thresh_bools):
                    model.compile(optimizer = 'adam', loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])
                    #model.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'])
                    results= []
                    #train_count += 1
                    training_directions = [i for i in range(len(directions)) if directions[i] in movable_directions and thresh_bools[i]==0]
                    if len(training_directions) == 0:
                        raise TypeError('train_directions is empty')
                    else:
                        print('Training_directions: ' + str(training_directions))
                        for direction in training_directions:
                            try:
                                print('using fit')
                                result = model.fit(np.array([grid1.tensor]), np.array([direction]), epochs = 1)
                            except:
                                for i in dir():
                                    print('Size of ' + str(i) + ': ' + str(sys.getsizeof(i)))
                                from psutil import Process
                                Process().memory_info().rss / (1024 * 1024)
                                snapshot = tracemalloc.take_snapshot()
                                top_stats = snapshot.statistics('lineno')
                                print('peak memory: ' + str(Process().memory_info().peak_wset))
                                print('score: ' + str(score))
                                print('train_counts: ' + str(sum(train_counts)))
                                print(f'train_counts: {sum(train_counts)}')
                                raise TypeError('something went wrong with fit')
                                result = None
                            print('fit is good')
                            results.append(result)
                            train_count += 1
                        predictions = model(np.array([grid1.tensor]))
                        predictions1 = predictions.numpy()
                        thresh_bools = make_thresh_bools(predictions1, training_threshold, movable_directions)
                        print('predictions: ' + str(predictions1[0]))
                        print('thresh_bools: ' + str(thresh_bools))
                        #nowwhat = input('again?')
                        #if nowwhat == 'n':
                        #    raise TypeError('Oopsies!')
        model.save('Model_makes_possible_moves_only')
        print('Game # ' + str(i) + ' is Done!')
        train_counts.append(train_count)
    print(train_counts)
    print('Total trains: ' + str(sum(train_counts)))
    #plot outcome
    #names = [i for i in range(len(train_counts))]
    #values = train_counts
    #fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    #axs[0].bar(names, values)
    #axs[1].plot(names, values)
    #plt.show()





#for evolving a good model
# Playing and Training loop
#options for bad_move_response: 'train', 'kill', 'mutate'
bad_move_response = 'kill'
def EvolveModel(bad_move_response = 'kill'):
#    pass
#if True:
    training_dict[0] = model_dict
    for iter_num in range(training_dict['numIterations']):     
        max_score = 0
        best_model = np.nan
        best_layer_count = 0
        # Have each model play the game
        model_dict = training_dict[iter_num]
        for i in model_dict:
            # Let the model update the grid until it can't move
            grid1 = Grid()
            model_dict[i]['isKilled'] = False
            model = model_dict[i]['model']
            score = 0
            while grid1.isMovable():
                predictions = model(np.array([grid1.tensor]))
                predictions1 = predictions.numpy()
                max1 = predictions1.max()
                ind1 = np.where(predictions1[0] == max1)[0][0]
                print('Predictions: ' + str(predictions1))
                print('Directions : [[___left___ __right__ ____up____ ___down___]]') 
                print('Chosen Direction: ' +str(directions[ind1]))
                model_dict[i]['predictions'] = predictions1 
                model_dict[i]['direction'] = directions[ind1] 
                if grid1.isMovable(directions[ind1]):
                    grid1.update_grid(directions[ind1])
                    #record the score
                    score = grid1.calculate_score()
                    model_dict[i]['score'] = score

                elif bad_move_response == 'train':
                    if training_mode == 'old': # 5 epochs in a directions until the program chooses that direction once. Do that for each possible direction to move
                        model.compile(optimizer = 'adam', loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])
                        for direction in directions:
                            if grid1.isMovable(direction):
                                isAccurate = False
                                while True:
                                    result = model.fit(np.array([grid1.tensor]), np.array([directions.index(direction)]), epochs=5)
                                    if any(result.history['accuracy']):
                                        print('break')
                                        break
                        print(direction + ' is done!')
                    elif training_mode == 'new': # 1 epoch for each direction it can move until at least on movable direction is chosen.
                        model.compile(optimizer = 'adam', loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])
                        movable_directions = [direction for direction in directions if grid1.isMovable(direction)]
                        while True:
                            results= []
                            for direction in movable_directions:
                                results.append(model.fit(np.array([grid1.tensor]), np.array([directions.index(direction)]), epochs = 1))
                            if any([any(result.history['accuracy']) for result in results]):
                                print('break on ' + direction)
                                break
                    elif training_mode == 'thresh_90':
                        model.compile(optimizer = 'adam', loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])
                        movable_directions = [direction for direction in directions if grid1.isMovable(direction)]
                        while True:
                            results= []
                            for direction in movable_directions:
                                results.append(model.fit(np.array([grid1.tensor]), np.array([directions.index(direction)]), epochs = 1))
                            if any([any(result.history['accuracy']) for result in results]):
                                print('break on ' + direction)
                                break


                elif bad_move_response == 'kill':
                    training_dict[iter_num]['isKilled'] = True
                    break
                elif bad_move_response == 'mutate':
                    pass #add a mutate

                else:
                    raise InputError('bad_move_response is not set to something that we use') 
            #keep the best model
            best_model = model if score > max_score else best_model
            max_score = score if score > max_score else max_score
            best_layer_count  = model_dict[i]['Dense_Layers'] if score > max_score else best_layer_count
        model_dict['max_score'] = max_score
        #Next, clone, then mutate the best model to make several new models
        best_model.save(best_models_folder+ '/best_model_'+str(iter_num)+'_'+str(max_score))
        #model_dict['best_model'] = deepcopy(best_model) this didnt work
        model_dict['best_layer_count'] = best_layer_count
        training_dict[iter_num] = model_dict
        print('Round #' + str(iter_num) + ' Scored: ' + str(training_dict[iter_num]['max_score']))
        #reset model_dict for the next round including cloning and mutating the best model
        model_dict = generate_model_dict(load_model_path = get_best_model_path())
        training_dict[iter_num + 1] = model_dict
        print('Iter_num: '+str(iter_num))
        #Repeat
        



print(str(args))
if 'Train_No_Bad_Direction' in args:
    Train_No_Bad_Direction()














#        with tf.GradientTape() as tape:
#            # Forward pass.

            
            # Compute the loss value for this batch.
            #loss_value = loss_fn(targets, predictions)

    # Get gradients of loss wrt the weights.
    #gradients = tape.gradient(loss_value, model.trainable_weights)
    # Update the weights of the model.
    #optimizer.apply_gradients(zip(gradients, model.trainable_weights))

