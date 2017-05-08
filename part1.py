from collections import deque
from copy import deepcopy
import numpy as np
import random
import csv


# Function definitions
def permute2(f, m):
    copyf = deepcopy(f);
    a = np.random.randint(low = 0, high = m)
    b = np.random.randint(low = 0, high = m)
    
    vala = f[a];
    valb = f[b]
    
    copyf[a] = valb;
    copyf[b] = vala;
    
    return copyf;

# Return the index in the origional alphabet of the symbol 
def f_inv(f, yi):
    return f.index(yi)

# Calculate the un-normalized posterior, p(f|y)
def posterior(f, ciphertext, state_probs, trans_probs):
    # Don't include the final newline symbol
    x_vector = [f_inv(f, yi) for yi in ciphertext if yi != '\n'];
    
    post = np.log2(state_probs[x_vector[0]])
    for i in xrange(len(x_vector)):
        if x_vector[i] == '\n': continue;
        if i == 0: continue
        if trans_probs[x_vector[i], x_vector[i-1]] == 0.0:
            return -float('inf');
        post += np.log2(trans_probs[x_vector[i], x_vector[i-1]])
        if post == -float('inf'): break;
    return post

def calc_accuracy(f, ciphertext, plaintext, alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '.']):
    x_vector = [f_inv(f, yi) for yi in ciphertext if yi != '\n'];
    x_vector = [alphabet[x] for x in x_vector]
    y_vector = [y for y in plaintext if y != '\n'];
    
    matches = 0;
    for i, letter in enumerate(y_vector):
        if letter == x_vector[i]:
            matches += 1;

    return float(matches) / float(len(y_vector));

# The f passed in here should always be the identity alphabet mapping
def calc_start(f, ciphertext, m = 28):
    state_probs = np.zeros((m,));
    trans_probs = np.zeros((m, m));
   
    # Given the alphabet mapping f, this just converts to integers
    x_vector = [f_inv(f, yi) for yi in ciphertext if yi != '\n'];
   

    for i, letter in enumerate(x_vector):
        state_probs[letter] += 1;
        if i > 0:
            trans_probs[letter, x_vector[i-1]] += 1;

    trans_probs = np.divide(trans_probs, state_probs)
    state_probs /= len(x_vector)

    print np.sum(state_probs)
    print np.sum(trans_probs, axis = 0)


def decode(ciphertext, output_file_name):
    # Read in datasets
    alphabet = [];
    with open('./alphabet.csv') as csvfile:
        filereader = csv.reader(csvfile)
        for row in filereader:
            alphabet = row
            
    m = len(alphabet);        
            
    state_probs = np.zeros((m,));
    trans_probs = np.zeros((m, m));

    with open('./letter_probabilities.csv') as csvfile:
        filereader = csv.reader(csvfile)
        for row in filereader:
            state_probs[:] = row;
            
    with open('./letter_transition_matrix.csv') as csvfile:
        filereader = csv.reader(csvfile)
        for i, row in enumerate(filereader):
            trans_probs[i, :] = row;

    f_curr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '.']
    calc_start(f_curr, ciphertext, m)
            
    # Start MH Algorithm
    MAX_ITERS = 3000;
    WINDOW = 100;

    #f_curr = ['s', 'g', 'p', 'c', 'b', 'z', 'd', 'i', 'w', 'n', 'o', 'k', 'r', 'e', '.', 'q', 'm', 'a', ' ', 'x', 'h', 't', 'f', 'v', 'l', 'u', 'j', 'y']
    random.shuffle(f_curr);

    log_likelihood = [];
    symbol_log_likelihood = [];

    acceptence = deque();
    accept_ratio = [];
    accuracy_ratio = [];
    accept_time = 0;

    iter = 0;
    count = 0;

    while iter < MAX_ITERS: 
        f_next = permute2(f_curr, m)
        
        next_post = posterior(f_next, ciphertext, state_probs, trans_probs)
        current_post = posterior(f_curr, ciphertext, state_probs, trans_probs)

        if current_post == -float('inf'):
            accept_rate = 1.0;
            iter -= 1; # Don't count interations until we get to a possible state
        else:
            accept_rate = min(1.0, np.exp(next_post - current_post))
            
        u = np.random.uniform();
        accept = 0;
        if u <= accept_rate:
            f_curr = f_next;
            like = next_post
            log_likelihood.append(next_post);
            symbol_log_likelihood.append(next_post/len(ciphertext));
            accept = 1; 
        else:
            f_curr = f_curr;
            like = current_post
            log_likelihood.append(current_post);
            symbol_log_likelihood.append(current_post/len(ciphertext));

        acceptence.append(accept);

        # Don't have access to plaintext tradiationally
        #accuracy = calc_accuracy(f_curr, ciphertext, plaintext, alphabet):
        #accuracy_ratio.append(accuracy);
        #if accuracy > 0.98:
        #    break;

        accept_time += 1;
        if accept_time > WINDOW:
            acceptence.popleft();
            accept_ratio.append(float(acceptence.count(1))/ float(WINDOW))
        
        print "(", count, ") Iterations:", iter, "Likelihood:", like 
        iter += 1; count += 1;

    x_vector = [f_inv(f_curr, yi) for yi in ciphertext if yi != '\n'];
    x_vector = [alphabet[x] for x in x_vector]

    f = open(output_file_name, 'w')
    f.write(''.join(x_vector))
    f.close()
    
    return f_curr

