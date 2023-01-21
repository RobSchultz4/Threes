from subprocess import run, PIPE

import Threes_ML_generator # To decrease the start up time for each run call

import keyboard




#command = ['python', 'Train_ML_test.py']
#Train ML to not choose a direction the game can't go

def run_command(command, num_games):
    p1s = []
    for i in range(num_games):
        print('running: ' + ' '.join(command))
        p1 = run(command, capture_output = True)
        final_score = [j for j in str(p1.stdout).split('\\r\\n') if any([i in j for i in ['Score']])][-1]
        print('run number: ' + str(i) + ' finished with a score of: ' + str(final_score))
        p1s.append(p1)
        event = keyboard.read_event()
        if event.event_type == keyboard.KEY_DOWN:
            if event.name =='space':
                break
            if event.name in 'poiuytrewqlkjhgfdsamnbvcxz':
                print('\n'.join([j for j in str(p1.stdout).split('\\r\\n') if any([i in j for i in ['=']])]))
                print('\n'.join([j for j in str(p1.stdout).split('\\r\\n') if any([i in j for i in ['Chosen']])]))
                print('\n'.join([j for j in str(p1.stdout).split('\\r\\n') if any([i in j for i in ['Training']])]))
                print('\n'.join([j for j in str(p1.stdout).split('\\r\\n') if any([i in j for i in ['Predictions']])]))
                print('\n'.join([j for j in str(p1.stdout).split('\\r\\n') if any([i in j for i in ['Score']])]))
                #print('\n'.join([j for j in str(p1.stdout).split('\\r\\n') if any([i in j for i in ['Predictions','Directions','Chosen']])]))
                print('\n'.join([j for j in str(p1.stdout).split('\\r\\n') if any([i in j for i in ['train']])]))

    return p1s

'''
if __name__ == '__main__':
    num_games = 1
    command = ['python', 'Threes_ML_generator.py','Train_No_Bad_Direction' ]
    run_command(command, num_games)
'''     





