# Import libraries
import os
import numpy as np
from lib.xror import XROR
from lib.parse import handle_bsor
from tqdm import tqdm, trange

# Configuration options
MIN_REPLAYS = 250   # Minimum number of replays per user for inclusion
MAX_REPLAYS = 999   # Maximum number of replays per user for inclusion
NUM_OPTIONS = 5     # Total number of options to choose from
SEARCH_SIZE = 99    # Maximum number of maps to search for comparison
TEST_NUMBER = 10    # Total number of tests to give participants
SCORE_DELTA = 0.01  # Maximum percentage score difference of replays

# Set fixed random seed
np.random.seed(0)

# Get list of replays
with open('./all-replays.txt') as file:
    replays = file.readlines()

# Remove extensions
replays = [r.split('.')[0] for r in replays]

# Sort by map
print("Sorting replays...")
maps = {}
for r in tqdm(replays):
    map = r.split('-', 1)[-1]
    if (not map in maps): maps[map] = []
    maps[map].append(r)

# Group replays by users
print("Clustering users...")
users = {}
user_names = []
for name in tqdm(replays):
    user = name.split('-')[0]
    if (len(user) > 17): continue

    if not user in users:
        users[user] = []
        user_names.append(user)
    users[user].append(name)

# Remove users with too few or too many replays
print("Filtering users...")
for user in tqdm(user_names.copy()):
    if (len(users[user]) < MIN_REPLAYS or len(users[user]) > MAX_REPLAYS):
        del users[user]
        user_names.remove(user)

# Make a single anonymous replay
def get_replay(name, source):
    with open('Z:/beatleader/replays/' + source + '.bsor', 'rb') as f:
        xror = XROR.fromBSOR(f)
    frames = handle_bsor('Z:/beatleader/replays/' + source + '.bsor')
    times = np.arange(0, 30, 1/30).reshape(900,1)

    xror.data['info']['software']['activity']['failTime'] = 30
    xror.data['info']['timestamp'] = 0
    xror.data['info']['hardware']['devices'][0]['name'] = 'HMD'
    xror.data['info']['hardware']['devices'][1]['name'] = 'LEFT'
    xror.data['info']['hardware']['devices'][2]['name'] = 'RIGHT'
    xror.data['info']['software']['app']['version'] = '1.X.X'
    xror.data['info']['software']['app']['extensions'][0]['version'] = '0.X.X'
    xror.data['info']['software']['runtime'] = 'oculus'
    xror.data['info']['software']['api'] = 'Oculus'
    xror.data['info']['user']['id'] = '100000000'
    xror.data['info']['user']['name'] = 'Anonymous'
    xror.data['frames'] = np.hstack([times, frames])

    bsor = xror.toBSOR()
    with open('./data/replays/' + name + '.bsor', 'wb') as f:
        f.write(bsor)

    return xror.data['info']['software']['activity']['score']

# Percentage difference
def pd(n1, n2):
    return abs(n1 - n2) / ((n1 + n2) / 2)

# Make a single challenge
def get_challenge(i):
    # Create output folder
    if not os.path.exists('./data/replays/' + i):
        os.makedirs('./data/replays/' + i)

    # Pick target user at random
    target_user = np.random.choice(user_names)

    # Shuffle labels
    labels = list(range(NUM_OPTIONS))
    np.random.shuffle(labels)

    # Pick reference replay at random
    reference_replay = np.random.choice(users[target_user]).strip()
    get_replay(i + '/ref', reference_replay)

    # Pick target replay at random
    target_replay = np.random.choice(users[target_user]).strip()
    target_map = target_replay.split('-', 1)[-1]
    if (target_replay == reference_replay): raise Exception()
    target_score = get_replay(i + '/' + str(labels[-1]), target_replay)

    # Pick target comparisons at random
    replay_pool = []
    for j in trange(NUM_OPTIONS - 1, position=1, leave=False, desc='Samples: '):
        for t in trange(SEARCH_SIZE + 1, position=2, leave=False, desc='Attempt: '):
            if t == SEARCH_SIZE: raise Exception()
            replay = np.random.choice(maps[target_map]).strip()
            if (replay == target_replay): continue
            if (replay in replay_pool): continue
            user = replay.split('-')[0]
            if (user not in user_names): continue
            replay_pool.append(replay)
            try:
                replay_score = get_replay(i + '/' + str(labels[j]), replay)
            except:
                continue
            if (pd(replay_score, target_score) > SCORE_DELTA): continue
            break

    # Return the answer
    return labels[-1]

# Create a set of challenges
print("Making challenges...")
answer_key = []
for n in trange(TEST_NUMBER, position=0, desc='Targets: '):
    while (len(answer_key) <= n):
        try:
            answer = get_challenge(str(n))
            answer_key.append(str(answer))
        except:
            pass

with open('./data/answer_key.txt', 'w') as f:
    f.write(','.join(answer_key))
