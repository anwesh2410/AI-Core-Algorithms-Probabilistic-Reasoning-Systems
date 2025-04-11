import numpy as np
from queue import PriorityQueue
import time

def read_vocabulary(file_path):
    with open(file_path, 'r') as f:
        vocabulary = f.read().splitlines()
    return vocabulary

def read_transition_matrix(file_path , L):
    transition_matrix = np.loadtxt(file_path)

    for i in range(L):
        row_sum = np.sum(transition_matrix[i])
        total_sum = row_sum + transition_matrix[L+1][i]
        transition_matrix[i, :L] /= total_sum
        transition_matrix[L+1, i] /= total_sum

    transition_matrix[L, :L] /= np.sum(transition_matrix[L, :L])

    return transition_matrix


def iddfs_sentence_generator(L, n, transition_matrix, vocabulary):
    node_count = 0  

    def dfs(depth, path, score, max_depth):
        nonlocal best_path, best_score, node_count
        node_count += 1  

        if depth == n:
            end_score = score * transition_matrix[L+1, path[-1]]
            if end_score > best_score:
                best_score = end_score
                best_path = path + [L+1] 
            return

        if depth == max_depth:
            return

        for word_index in range(L):
            if depth == 0:
                new_score = score * transition_matrix[L, word_index]  
            else:
                new_score = score * transition_matrix[path[-1], word_index]  
            
            dfs(depth + 1, path + [word_index], new_score, max_depth)

    best_path = []
    best_score = float('-inf')

    # IDDFS loop: progressively increase the depth
    for max_depth in range(1, n + 1):
        dfs(0, [], 1, max_depth)  # Start DFS with score 1 from <SoS>

    optimal_sentence = ['<SoS>'] + [vocabulary[i] for i in best_path[:-1]] + ['<EoS>']
    return optimal_sentence, best_score, node_count

def iddfs(L , n , transition_matrix, vocabulary):

    start_time = time.time()
    optimal_sentence, score, node_count = iddfs_sentence_generator(L, n, transition_matrix, vocabulary)

    end_time = time.time()
    compute_time = end_time - start_time

    print("----IDDFS----")
    print("Optimal Sentence:", " ".join(optimal_sentence))
    print(f"Score: {score:}")
    print(f"Nodes Explored: {node_count}")
    print(f"Compute Time: {compute_time:.6f} seconds")
    print("------------")


def ucs_sentence_generator(L, n, transition_matrix, vocabulary):
    pq = PriorityQueue()
    start_state = (1, -1, [])  
    pq.put((-1, start_state)) 
    
    best_prob = float('-inf')
    best_sentence = None
    nodecount = 0
    
    while not pq.empty():
        _, (prob, last_word, path) = pq.get()
        nodecount += 1
        
        if len(path) == n:
            end_prob = prob * transition_matrix[L+1, last_word]
            if end_prob > best_prob:
                best_prob = end_prob
                best_sentence = ['<SoS>'] + [vocabulary[i] for i in path] + ['<EoS>']
            continue  
        
        for next_word in range(L):
            if len(path) == 0:
                new_prob = prob * transition_matrix[L, next_word]
            else:
                new_prob = prob * transition_matrix[last_word, next_word]
            
            new_path = path + [next_word]
            new_state = (new_prob, next_word, new_path)
            pq.put((-new_prob, new_state))
    
    return best_sentence, best_prob, nodecount


def UCS(L , n , transition_matrix, vocabulary):

    start_time = time.time()
    optimal_sentence, score, node_count = ucs_sentence_generator(L, n, transition_matrix, vocabulary)

    end_time = time.time()
    compute_time = end_time - start_time

    print("----UCS----")
    print("Optimal Sentence:", " ".join(optimal_sentence))
    print(f"Score: {score:}")
    print(f"Nodes Explored: {node_count}")
    print(f"Compute Time: {compute_time:.6f} seconds")
    print("------------")


def heuristic(L, current_length, n, transition_matrix):
    best_transition = np.max(transition_matrix[:L, :L])  
    remaining_length = n - current_length
    return np.log(best_transition) * remaining_length  


def greedy_search_sentence_generator(L, n, transition_matrix, vocabulary):
    path = []
    current_word = -1  
    probability = 1.0
    nodecount = 0

    for i in range(n):
        if i == 0:
            candidate_scores = transition_matrix[L] + heuristic(L, i, n, transition_matrix)
            next_word = np.argmax(candidate_scores)
        else:
            candidate_scores = transition_matrix[current_word] + heuristic(L, i, n, transition_matrix)
            next_word = np.argmax(candidate_scores)
        
        path.append(next_word)
        if i == 0:
            probability *= transition_matrix[L, next_word]
        else:
            probability *= transition_matrix[current_word, next_word]
        current_word = next_word
        
        nodecount += 1
    probability *= transition_matrix[L+1, current_word]

    sentence = ['<SoS>'] + [vocabulary[i] for i in path] + ['<EoS>']
    return sentence, probability , nodecount


def greedy(L , n , transition_matrix, vocabulary):

    start_time = time.time()
    optimal_sentence, score, node_count = greedy_search_sentence_generator(L, n, transition_matrix, vocabulary)

    end_time = time.time()
    compute_time = end_time - start_time

    print("----Greedy----")
    print("Optimal Sentence:", " ".join(optimal_sentence))
    print(f"Score: {score:}")
    print(f"Nodes Explored: {node_count}")
    print(f"Compute Time: {compute_time:.6f} seconds")
    print("------------")

def astar_sentence_generator(L, n, transition_matrix, vocabulary):
    pq = PriorityQueue()
    start_state = (0, L, [])  
    f = -heuristic(L, 0, n, transition_matrix)  
    pq.put((f, start_state))
    nodecount = 0

    best_sentence = None
    best_score = float('-inf')

    while not pq.empty():
        _, (log_prob, last_word, path) = pq.get()
        nodecount += 1

        if len(path) == n:
            end_log_prob = log_prob + np.log(transition_matrix[L+1, last_word])
            score = np.exp(end_log_prob)
            if score > best_score:
                best_score = score
                best_sentence = ['<SoS>'] + [vocabulary[i] for i in path] + ['<EoS>']
            continue

        for next_word in range(L):
            new_log_prob = log_prob + np.log(transition_matrix[last_word, next_word])
            new_path = path + [next_word]
            h = heuristic(L, len(new_path), n, transition_matrix)
            f = -(new_log_prob + h)  
            new_state = (new_log_prob, next_word, new_path)
            pq.put((f, new_state))

    return best_sentence, best_score , nodecount

def astar(L , n , transition_matrix, vocabulary):

    start_time = time.time()
    optimal_sentence_astar, score_astar, node_count_astar = astar_sentence_generator(L, n, transition_matrix, vocabulary)

    end_time = time.time()
    compute_time = end_time - start_time

    print("----A-Star----")
    print("Optimal Sentence:", " ".join(optimal_sentence_astar))
    print(f"Score: {score_astar:}")
    print(f"Nodes Explored: {node_count_astar}")
    print(f"Compute Time: {compute_time:.6f} seconds")
    print("------------")

def modified_iddfs_sentence_generator(L, n, transition_matrix, vocabulary):
    node_count = 0  

    def dfs(depth, path, score):
        nonlocal best_path, best_score, found_solution, node_count
        node_count += 1  

        if score <= best_score:
            return

        if depth == n:
            end_score = score * transition_matrix[L+1, path[-1]]  
            if end_score > best_score:
                best_score = end_score
                best_path = path + [L+1] 
                found_solution = True
            return

        for word_index in range(L):
            if depth == 0:
                new_score = score * transition_matrix[L, word_index]  
            else:
                new_score = score * transition_matrix[path[-1], word_index]  

            if new_score > best_score:
                dfs(depth + 1, path + [word_index], new_score)

    best_path = []
    best_score = float('-inf')
    found_solution = False
    
    for max_depth in range(1, n + 1):
        dfs(0, [], 1)  
        if found_solution:
            break  
    
    optimal_sentence = ['<SoS>'] + [vocabulary[i] for i in best_path[:-1]] + ['<EoS>']
    return optimal_sentence, best_score, node_count

def iddfsmodified(L , n , transition_matrix, vocabulary):

    start_time = time.time()
    optimal_sentence, score, node_count = modified_iddfs_sentence_generator(L, n, transition_matrix, vocabulary)

    end_time = time.time()
    compute_time = end_time - start_time

    print("----MODIFIED IDDFS----")
    print("Optimal Sentence:", " ".join(optimal_sentence))
    print(f"Score: {score:}")
    print(f"Nodes Explored: {node_count}")
    print(f"Compute Time: {compute_time:.6f} seconds")
    print("------------")


def main():
    L, n_plus_2 , transition_matrix_file, vocabulary_file = input().split()
    L = int(L)
    n = int(n_plus_2) - 2  

    vocabulary = read_vocabulary(vocabulary_file)
    transition_matrix = read_transition_matrix(transition_matrix_file, L)


    iddfs(L , n , transition_matrix , vocabulary)
    UCS(L , n , transition_matrix , vocabulary)
    greedy(L , n , transition_matrix , vocabulary)
    astar(L , n , transition_matrix , vocabulary)
    iddfsmodified(L , n , transition_matrix , vocabulary)
    
if __name__ == "__main__":
    main()