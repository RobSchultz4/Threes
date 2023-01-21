# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 01:04:32 2022

@author: thesc
"""

# commandline threes game 


import numpy as np
import pandas as pd
import keyboard
from random import choice
from random import shuffle
from copy import copy
import random


def isArrowPressed(event):
    return any([event.name == i for i in ['up','down','left','right']])



def save_score(score):
    print('Final Score: ' + str(score))
    print('Top Scores coming soon')


def exit_to_menu():
    quit()
    


#k = 1
#for i in grid.index:
#    for j in grid.columns:
#        grid.loc[i,j] = i+j
#        
#        grid.loc[i,j] = k
#        k += 2                #

#grid.loc[rownum, colnum]

class Grid():
    def __init__(self):
        #initialize the grid
        first_ten = [1,1,1,2,2,2,3,3,3,choice([1,2,3])]
        shuffle(first_ten)
        first_grid_nums = [0]*7 + first_ten[0:-1]
        shuffle(first_grid_nums)
        first_row = first_grid_nums[0:4]
        second_row = first_grid_nums[4:8]
        third_row = first_grid_nums[8:12]
        fourth_row = first_grid_nums[12:16]         #
        self.grid = pd.DataFrame([first_row, second_row,third_row,fourth_row])
        self.calc_moves()
        self.score = 0
        self.next_ten = [random.randrange(1,4,1) for i in range(0,10)]
        self.next_tile = first_ten[-1]
        self.score_rubric = pd.Series([0,0,0] + [3**i for i in range(1,14,1)],index = [0,1,2,3,6,12,24,48,96,192,384,768,1536,3072,6144,12288])
        self.flatten = first_row + second_row + third_row + fourth_row + [self.next_tile]
        self.tensor = np.array([self.grid.iloc[i] for i in range(4)] + [np.array([self.next_tile,0,0,0])])
        #self.update_grid(choice(['up','down','left','right']))
    def game_over(self):
        score = self.calculate_score()
        print('Game Over')
        print('Final Score: ' + str(score))
        save_score(score)
        exit_to_menu()

    def display_grid(self):
        print(self.moves_grid)
        print(str(self.grid))
        print('Next Piece: ' + str(self.next_tile))
        print('Score: ' + str(self.calculate_score()))
        
    def calculate_score(self):
        tiles1 = []
        tile_points = []
        for i in self.grid.columns:
            for j in self.grid.index:
                if self.grid.loc[i,j] > 2:
                    tiles1.append(self.grid.loc[i,j])
        for i in tiles1:
            tile_points.append(self.score_rubric[i])
        score = sum(tile_points)
        return score
    
    def calc_moves(self):
        moves_grid = pd.DataFrame([[[]]*4]*4)
        for i in self.grid.index:
            for j in self.grid.columns:
                # Check that zeros aren't all behind it 
                if self.grid.loc[i,j] == 0:
                    if any(not k==0 for k in self.grid.columns[i:]):
                        moves_grid.loc[i,j] = moves_grid.loc[i,j] + ['down','left','right']
                    if any(not k==0 for k in self.grid.columns[:i]):
                        moves_grid.loc[i,j] = moves_grid.loc[i,j] + ['up','left','right']
                    if any(not k==0 for k in self.grid.index[j:]):
                        moves_grid.loc[i,j] = moves_grid.loc[i,j] + ['up','down','right']
                    if any(not k==0 for k in self.grid.index[:j]):
                        moves_grid.loc[i,j] = moves_grid.loc[i,j] + ['up','down','left']
               
                # Check for 1s
                if self.grid.loc[i,j] == 1 and any(2 == k for k in [self.grid.loc[i+1,j] if i+1<=3 else 0, self.grid.loc[i-1,j] if i-1>=0 else 0]):
                    moves_grid.loc[i,j] = moves_grid.loc[i,j] + ['up', 'down']
                if self.grid.loc[i,j] == 1 and any(2 == k for k in [self.grid.loc[i,j+1] if j+1<=3 else 0, self.grid.loc[i,j-1] if j-1>=0 else 0]):
                    moves_grid.loc[i,j] = moves_grid.loc[i,j] + ['left', 'right']

                #Check for 2s
                if self.grid.loc[i,j] == 2 and any(1 == k for k in [self.grid.loc[i+1,j] if i+1<=3 else 0, self.grid.loc[i-1,j] if i-1>=0 else 0]):
                    moves_grid.loc[i,j] = moves_grid.loc[i,j] + ['up', 'down']
                if self.grid.loc[i,j] == 2 and any(1 == k for k in [self.grid.loc[i,j+1] if j+1<=3 else 0, self.grid.loc[i,j-1] if j-1>=0 else 0]):
                    moves_grid.loc[i,j] = moves_grid.loc[i,j] + ['left', 'right']             

                #Check for 3s or higher
                if self.grid.loc[i,j] >= 3 and self.grid.loc[i,j] == self.grid.loc[i+1,j] if i+1<=3 else np.nan:
                    moves_grid.loc[i,j] = moves_grid.loc[i,j] + ['up']
                if self.grid.loc[i,j] >= 3 and self.grid.loc[i,j] == self.grid.loc[i-1,j] if i-1>=0 else np.nan:
                    moves_grid.loc[i,j] = moves_grid.loc[i,j] + ['down']
                if self.grid.loc[i,j] >= 3 and self.grid.loc[i,j] == self.grid.loc[i,j+1] if j+1<=3 else np.nan:
                    moves_grid.loc[i,j] = moves_grid.loc[i,j] + ['left']
                if self.grid.loc[i,j] >= 3 and self.grid.loc[i,j] == self.grid.loc[i,j-1] if j-1>=0 else np.nan:
                    moves_grid.loc[i,j] = moves_grid.loc[i,j] + ['right']
                
                # Check if there is a zero next to it
                if any(0 == k for k in [self.grid.loc[i+1,j] if i+1<=3 else 1, self.grid.loc[i-1,j] if i-1>=0 else 1]):
                    moves_grid.loc[i,j] = moves_grid.loc[i,j] + ['up', 'down']
                if any(0 == k for k in [self.grid.loc[i,j+1] if j+1<=3 else 1, self.grid.loc[i,j-1] if j-1>=0 else 1]):
                    moves_grid.loc[i,j] = moves_grid.loc[i,j] + ['right', 'left']
               
                # Keep unique items only
                moves_grid.loc[i,j] = list(set(moves_grid.loc[i,j]))
                
                # Remove directions from edges since they cant move that direction.
                if i == 3 and 'up' in moves_grid.loc[i,j]:
                    moves_grid.loc[i,j].remove('up')
                if i == 0 and 'down' in moves_grid.loc[i,j]:
                    moves_grid.loc[i,j].remove('down')
                if j == 3 and 'left' in moves_grid.loc[i,j]:
                    moves_grid.loc[i,j].remove('left')
                if j == 0 and 'right' in moves_grid.loc[i,j]:
                    moves_grid.loc[i,j].remove('right')
                
                # Remove anything pulling a zero
                if 'up' in moves_grid.loc[i,j] and not i == 3 and self.grid.loc[i+1,j] == 0:
                    moves_grid.loc[i,j].remove('up')
                if 'down' in moves_grid.loc[i,j] and not i == 0 and self.grid.loc[i-1,j] == 0:
                    moves_grid.loc[i,j].remove('down')
                if 'left' in moves_grid.loc[i,j] and not j == 3 and self.grid.loc[i,j+1] == 0:
                    moves_grid.loc[i,j].remove('left')
                if 'right' in moves_grid.loc[i,j] and not j == 0 and self.grid.loc[i,j-1] == 0:
                    moves_grid.loc[i,j].remove('right')

                '''
                if i==3 and not self.grid.loc[i,j] == 0 and 'down' in moves_grid.loc[i,j]:
                    moves_grid.loc[i,j].remove('down')
                if i==0 and not self.grid.loc[i,j] == 0 and 'up' in moves_grid.loc[i,j]:
                    moves_grid.loc[i,j].remove('up')
                if j==3 and not self.grid.loc[i,j] == 0 and 'right' in moves_grid.loc[i,j]:
                    moves_grid.loc[i,j].remove('right')
                if j==0 and not self.grid.loc[i,j] == 0 and 'left' in moves_grid.loc[i,j]:
                    moves_grid.loc[i,j].remove('left')
                '''
        self.moves_grid = moves_grid

    def isMovable(self, direction = None):
        if direction == None:
            return any(self.moves_grid.any())
        else:
            return any(self.moves_grid.applymap(lambda x: direction in x).any())

    def find_next_tile(self):
        if not 1 in self.next_ten:
            pos1 = random.randrange(0,10,1)
            self.next_ten[pos1] = 1
        if not 2 in self.next_ten:
            pos1 = random.randrange(0,10,1)
            self.next_ten[pos1] = 2
        if not 3 in self.next_ten:
            pos1 = random.randrange(0,10,1)
            self.next_ten[pos1] = 3
        grid_vals = [j for i in self.grid.values for j in i]
        list123 = copy(self.next_ten)+grid_vals
        proportion1 = list123.count(1)/len(list123)
        proportion2 = list123.count(2)/len(list123)
        proportion3 = list123.count(3)/len(list123)
        last_tile = self.next_ten.pop(0)
        list1 = [1]*round(1/proportion1)
        list2 = [2]*round(1/proportion2)
        list3 = [3]*round(1/proportion3)
        weighted_probability_list= list1 + list2 + list3
        print('pro1: ' + str(proportion1))
        print('pro2: ' + str(proportion2))
        print('pro3: ' + str(proportion3))
        print('weighted_probability_list: '+ str(weighted_probability_list))
        self.next_ten.append(choice(weighted_probability_list))
        print('next_ten: ' + str( self.next_ten))
        return self.next_ten[0] 
        #return choice([1,2,3])

    def move_right(self):
        col_movements = pd.Series([False,False,False,False], index = self.grid.index)
        new_tile_placements = pd.Series([self.moves_grid.T[i].apply(lambda x: 'right' in x).any() for i in self.moves_grid.columns], index = self.moves_grid.columns)
        new_tile_placement = choice([i for i in new_tile_placements.index if new_tile_placements[i]])
        index= list(copy(self.grid.index))
        columns = list(copy(self.grid.index))
        columns.reverse()
        for j in columns:
            for i in index:
                # if the column moved and we're running the last row
                if col_movements[i] and not j-1 >= 0:
                    if i == new_tile_placement:
                        self.grid.loc[i,j] = self.next_tile
                    else:
                        self.grid.loc[i,j] = 0
                # if the column moved and we're running any row but the last row
                if col_movements[i] and j-1 >= 0:
                    self.grid.loc[i,j] = self.grid.loc[i,j-1]
                #if the column hasn't moved, and this spot can and we're not in the last row
                if not col_movements[i] and 'right' in self.moves_grid.loc[i,j] and j-1 >= 0:
                    self.grid.loc[i,j] = self.grid.loc[i,j] + self.grid.loc[i,j-1]
                    col_movements[i] = True

    def move_left(self):
        col_movements = pd.Series([False,False,False,False], index = self.grid.index)
        new_tile_placements = pd.Series([self.moves_grid.T[i].apply(lambda x: 'left' in x).any() for i in self.moves_grid.columns], index = self.moves_grid.columns)
        new_tile_placement = choice([i for i in new_tile_placements.index if new_tile_placements[i]])
        index= list(copy(self.grid.index))
        for j in self.grid.columns:
            for i in index:
                # if the column moved and we're running the last row
                if col_movements[i] and not j+1 <= 3:
                    if i == new_tile_placement:
                        self.grid.loc[i,j] = self.next_tile
                    else:
                        self.grid.loc[i,j] = 0
                # if the column moved and we're running any row but the last row
                if col_movements[i] and j+1 <= 3:
                    self.grid.loc[i,j] = self.grid.loc[i,j+1]
                #if the column hasn't moved, and this spot can and we're not in the last row
                if not col_movements[i] and 'left' in self.moves_grid.loc[i,j] and j+1 <= 3:
                    self.grid.loc[i,j] = self.grid.loc[i,j] + self.grid.loc[i,j+1]
                    col_movements[i] = True
    def move_up(self):
        col_movements = pd.Series([False,False,False,False], index = self.grid.index)
        new_tile_placements = pd.Series([self.moves_grid[i].apply(lambda x: 'up' in x).any() for i in self.moves_grid.columns], index = self.moves_grid.columns)
        new_tile_placement = choice([i for i in new_tile_placements.index if new_tile_placements[i]])
        for i in self.grid.index:
            for j in self.grid.columns:
                # if the column moved and we're running the last row
                if col_movements[j] and not i+1 <= 3:
                    if j == new_tile_placement:
                        self.grid.loc[i,j] = self.next_tile
                    else:
                        self.grid.loc[i,j] = 0
                # if the column moved and we're running any row but the last row
                if col_movements[j] and i+1 <= 3:
                    self.grid.loc[i,j] = self.grid.loc[i+1,j]
                #if the column hasn't moved, and this spot can and we're not in the last row
                if not col_movements[j] and 'up' in self.moves_grid.loc[i,j] and i+1 <= 3:
                    self.grid.loc[i,j] = self.grid.loc[i,j] + self.grid.loc[i+1,j]
                    col_movements[j] = True


    def move_down(self):
        col_movements = pd.Series([False,False,False,False], index = self.grid.index)
        new_tile_placements = pd.Series([self.moves_grid[i].apply(lambda x: 'down' in x).any() for i in self.moves_grid.columns], index = self.moves_grid.columns)
        new_tile_placement = choice([i for i in new_tile_placements.index if new_tile_placements[i]])
        index= list(copy(self.grid.index))
        index.reverse()
        for i in index:
            for j in self.grid.columns:
                # if the column moved and we're running the last row
                if col_movements[j] and not i-1 >= 0:
                    if j == new_tile_placement:
                        self.grid.loc[i,j] = self.next_tile
                    else:
                        self.grid.loc[i,j] = 0
                # if the column moved and we're running any row but the last row
                if col_movements[j] and i-1 >= 0:
                    self.grid.loc[i,j] = self.grid.loc[i-1,j]
                #if the column hasn't moved, and this spot can and we're not in the last row
                if not col_movements[j] and 'down' in self.moves_grid.loc[i,j] and i-1 >= 0:
                    self.grid.loc[i,j] = self.grid.loc[i,j] + self.grid.loc[i-1,j]
                    if not self.grid.loc[i,j] == 0:
                        col_movements[j] = True

    def update_grid(self, direction):
        if direction == 'right':
            self.move_right()
        if direction == 'left':
            self.move_left()
        if direction == 'up':
            self.move_up()
        if direction == 'down':
            self.move_down()
        '''
        new_grid = pd.DataFrame([[0]*4]*4)
        index = new_grid.index.to_list()
        columns = new_grid.columns.to_list()
        new_points = 0
        if direction == 'up':
            self.move_up()
            row_order = new_grid.index
            col_order = new_grid.columns
            di1, dj1 = 1,0
            iedge = row_order[-1]
            jedge = None
        if direction == 'down':
            index.reverse() 
            row_order = index 
            col_order = new_grid.columns
            di1, dj1 = -1,0
            iedge = row_order[-1] # good
            jedge = None # good
        if direction == 'left':
            row_order = new_grid.index
            col_order = new_grid.columns
            di1, dj1 = 0,1
            iedge = None
            jedge = col_order[-1]
        if direction == 'right':
            row_order = new_grid.index
            columns.reverse()
            col_order = columns
            di1, dj1 = 0,-1
            iedge = None
            jedge = col_order[-1]
        move_pos = []
        new_num_pos_order = new_grid.index.to_list()
        shuffle(new_num_pos_order)
        new_num_pos = None
        new_num_added = False
        for i in row_order:
            for j in col_order:
                #if direction == 'up':
                i1 = i + di1
                j1 = j + dj1
                print('###'*13)
                print('direction: '+ direction)
                print('i: ' + str(i))
                print('di1: ' + str(di1))
                print('i1: ' + str(i1))
                print('j: ' + str(j))
                print('dj1: ' + str(dj1))
                print('j1: ' + str(j1))
                print('move_pos before: ' + str(move_pos))
                #print('self.grid')
                #print(self.grid)
                #print(self.moves_grid)
                pos = j if jedge == None else i
                print('pos: '+ str(pos))
                if i == iedge or j == jedge:
                    if pos in new_num_pos_order and not pos in move_pos:
                        new_num_pos_order.remove(pos)
                    new_num_pos = new_num_pos_order[0] if new_num_pos == None and pos in move_pos else new_num_pos
                    if not new_num_added and pos == new_num_pos:
                        new_grid.loc[i,j] = self.next_tile 
                        new_num_added = True
                    else:
                        new_grid.loc[i,j] = 0
                    
                #elif j == jedge:
                #    if pos in new_num_pos_order and not pos in move_pos:
                #        new_num_pos_order.remove(pos)
                #    new_num_pos = choice(move_pos) if new_num_pos == None and j in move_pos else new_num_pos
                #    new_grid.loc[i,j] = self.next_tile if pos == new_num_pos else 0
                    #new_grid.loc[i,j] = self.next_tile
                elif pos in move_pos: # just pull the number below up to this slot
                    new_grid.loc[i,j] = self.grid.loc[i1,j1]
                elif ((self.grid.loc[i1,j1] == self.grid.loc[i,j] and self.grid.loc[i,j] > 2) or (1,2) == (self.grid.loc[i1,j1],self.grid.loc[i,j]) or (2,1) == (self.grid.loc[i1,j1],self.grid.loc[i,j]) or self.grid.loc[i,j] == 0):  # do the numbers combine?:
                    if (jedge == None and not j in move_pos) or (iedge == None and not i in move_pos):
                        new_grid.loc[i,j] = self.grid.loc[i1,j1] + self.grid.loc[i,j]
                    if not any(i == 0 for i in [self.grid.loc[i1,j1] , self.grid.loc[i,j]]):
                        new_points += self.grid.loc[i1,j1] + self.grid.loc[i,j]
                    move_pos.append(pos)
                else: 
                    new_grid.loc[i,j] = self.grid.loc[i,j]
                print('move_pos after: ' + str(move_pos))
                print('new_num_pos: ' + str(new_num_pos))
                print('new_grid: ')
                print(new_grid)
                print(move_pos)
        #self.grid = new_grid
        '''
        self.next_tile = self.find_next_tile()
        self.calc_moves()
        self.flatten = [self.grid.iloc[i] for i in range(4)] + [self.next_tile]
        self.tensor = np.array([self.grid.iloc[i] for i in range(4)] + [np.array([self.next_tile,0,0,0])])
        self.display_grid()

if __name__ =='__main__':
    grid1 = Grid()
    
    grid1.display_grid()
    while True:
        # Wait for the next event.
        event = keyboard.read_event()
        if event.event_type == keyboard.KEY_DOWN:
            print(event.name +' was pressed')
            if event.name == 't':
                print('next ten: ' + str(grid1.next_ten)) 
            if isArrowPressed(event) and grid1.isMovable(event.name):
                grid1.update_grid(event.name)
            elif not grid1.isMovable(event.name):
                print('Cant move ' + event.name)
    
        if event.event_type == keyboard.KEY_DOWN and event.name == 'space' or not grid1.isMovable():
            grid1.game_over()
            break
    
       
        













# Use Tensor Flow to make a deep learning network that owns this game

# Borrow a pretrained structure
# Change the last layer out for 4 options (up,down,left right)
# retrain the algorithm using cluster loss... IS there a way to make a graph that I can train it with? Maybe something simple (or maybe it's not so simple) like a graph the plots the ways to combine numbers into larger ones? Like Below:

## 1   2 2   1 
##  \ /   \ /
##   3     3
##    \   /
##      6
## Would then need to make a graph and learn how to make a graph but would probably be easier to just skip that for now unless the network sucks.

 









