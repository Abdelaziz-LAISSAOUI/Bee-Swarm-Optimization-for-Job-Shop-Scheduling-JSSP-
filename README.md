# 🐝 Bee Swarm Optimization for Job Shop Scheduling (JSSP)

This project implements the Bee Swarm Optimization (BSO) metaheuristic to solve the classic **Job Shop Scheduling Problem (JSSP)**. It is benchmarked against multiple instances from the **Taillard dataset** and includes tools for running large-scale experiments with grid search and CSV-based logging.

---

## 📦 Features

- ✅ Full BSO metaheuristic implementation
- ✅ Supports multiple benchmark instances (Taillard format)
- ✅ Operation-based encoding for solution representation
- ✅ Schedule decoder with makespan calculation and validity checks
- ✅ Grid search over BSO parameters
- ✅ Checkpoint logging (e.g., every 1000 iterations)
- ✅ Outputs results to `bso_results_20_15.csv`


---

## ⚙️ How It Works

1. **Input**: A benchmark file (like `tai20_15.txt`) containing multiple JSSP instances.
2. **Encoding**: Operation-based sequence of job IDs.
3. **BSO Logic**:
   - Initialize reference solution
   - Generate search area (neighbors)
   - Bees perform local search
   - Share best dance (solution)
   - Iterate and update
4. **Logging**: Results are stored in `bso_results.csv` with all parameters + makespan.


