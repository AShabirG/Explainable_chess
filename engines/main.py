from subprocess import Popen, PIPE, STDOUT, TimeoutExpired
import torch
import tensorflow as tf
from tensorflow.keras import Model
#maia_model = torch.load(r"C:\Users\Shabir\Documents\StockFish\model_files\ckpt-40-400000.pb")
maia_model = tf.keras.models.load_model(r"C:\Users\Shabir\Documents\StockFish\model_files\ckpt-40-400000.pb")
def ecommand(p, comm):
    p.stdin.write(f'{comm}\n')


def analyze(engine, board_state):
    suggested_move = '0000'

    p = Popen([engine], stdout=PIPE, stdin=PIPE, stderr=STDOUT, bufsize=0, text=True)

    ecommand(p, board_state) # input board state
    if engine == 'lc0.exe':
        ecommand(p, 'go nodes 1')
    else:
        ecommand(p, 'go nodes 600')

    for line in iter(p.stdout.readline, ''):  # read each line of engine output as replies from our command
        line = line.strip()

        if line.startswith('bestmove'):  # exit the loop when we get the engine bestmove
            suggested_move = line.split()[1].strip()
            break

    ecommand(p, 'quit')  # properly quit the engine

    # Make sure process 'p' is terminated (if not terminated for some reason) as we already sent the quit command.
    try:
        p.communicate(timeout=5)
    except TimeoutExpired:  # If timeout has expired and process is still not terminated.
        p.kill()
        p.communicate()

    return suggested_move


board = 'position startpos moves e2e4 c7c5'
maia = 'lc0.exe'
maia_move = analyze(maia, board)
stockfish = 'stockfish_15_win_x64_popcnt/stockfish_15_x64_popcnt.exe'
stockfish_move = analyze(stockfish, board)
print(f'Stockfish suggested move: {stockfish_move}')
print(f'Maia suggested move: {maia_move}')
