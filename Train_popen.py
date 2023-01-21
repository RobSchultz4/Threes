import Threes_ML_generator
from subprocess import run, PIPE

numGames = 1
#command = ['python', 'Threes_ML_generator.py','-Train_No_Bad_Direction' ]
command = ['python', 'Train_ML_test.py]
#Train ML to not choose a direction the game can't go
for _ in range(numGames):
    print('running: ' + ' '.join(command))
    subprocess.run(command, stdin = PIPE, stdout = PIPE, stderr =PIPE, universal_newLines = True)








