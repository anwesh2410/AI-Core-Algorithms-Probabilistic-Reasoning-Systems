
B22AI005_1 Assignment Submission

Files included in this zip:
1. B22AI005_1.py - Python script containing the implementation of various search algorithms (IDDFS, UCS, Greedy, A*, Modified IDDFS).
2. B22AI005_1.pdf - Report explaining the algorithms, results, and analysis.
3. B22AI005_1.txt - This README file explaining the content of the submission.

Input format:
   L n (path of transition file) (path of vocab file)

Input :
- The input consists of a vocabulary file and a transition matrix file, where L is the size of the vocabulary.
- The transition matrix represents probabilities for transitioning between words and includes special transitions for start-of-sentence (<SoS>) and end-of-sentence (<EoS>).
- Vocabulary file: Each line contains a word (e.g., word1, word2, etc.).
- Transition matrix file: Each row represents the probabilities of transitioning from one word to another.


The algorithms implemented in the script are:
   - IDDFS (Iterative Deepening Depth-First Search)
   - UCS (Uniform Cost Search)
   - Greedy Search
   - A* Search
   - Modified IDDFS (with pruning)

Run Instructions:
1. Run the Python script using the command: python B22AI005_1.py.

Expected Output:
- For each algorithm, the optimal sentence generated, the score (probability), the number of nodes explored, and the compute time are printed to the console.

Additional Information:
- The report (B22AI005_1.pdf) provides a detailed explanation of each algorithm, the experiment setup, and performance comparison.
