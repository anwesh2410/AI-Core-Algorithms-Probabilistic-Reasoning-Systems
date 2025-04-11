

Propositional Logic Theorem Provers

 Project Overview
This project implements two theorem proving algorithms for propositional logic: **Natural Deduction** and **Resolution Refutation**. The algorithms derive conclusions from a given knowledge base (KB) and demonstrate formal reasoning techniques.

 Objectives
- Implement natural deduction and resolution refutation algorithms.
- Analyze performance based on nodes explored and compute time.
- Validate algorithms with examples and analyze results.

 Algorithms
1. **Natural Deduction**: Uses inference rules (e.g., Modus Ponens) for deductions.
2. **Resolution Refutation**: Converts formulas to CNF and resolves clauses for contradictions.

 Implementation
- `natural_deduction.py`: Natural deduction implementation.
- `resolution_refutation.py`: Resolution refutation implementation.
- `examples/`: Contains test knowledge bases and queries.
- `results/`: Stores output files and performance metrics.

 Usage Instructions
1. Clone or download the repository.
2. Navigate to the project directory.
3. Run the algorithms:
   - Natural Deduction: `python natural_deduction.py`
   - Resolution Refutation: `python resolution_refutation.py`
4. Check `results/` for performance metrics and conclusions.

 Performance Analysis
- Tested with different KB sizes (n, m ∈ {5, 6, 7, 8}).
- Metrics include nodes explored and compute time.
