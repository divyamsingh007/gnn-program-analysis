# Graph Neural Network for Program Analysis: Graph-Based Vulnerability Detection Using AST, CFG and Data-Flow Representations

## 📖 Project Overview

This repository hosts a Compiler Design Lab research project focused on advancing source-code vulnerability detection through the application of **Graph Neural Networks (GNNs)**.

Traditional machine learning approaches for code analysis often treat source code as flat text or token sequences, ignoring the inherent structural and semantic relationships defined by the programming language. This project leverages the insight that programs naturally form complex, interconnected graphs during the compilation process. By utilizing **Clang and LLVM**, we extract deep, compiler-level representations of C/C++ source code and model them as multi-view graphs to train neural networks capable of identifying security vulnerabilities.

---

## 🎯 Research Objective

The primary research question driving this project is:

> *"Can combining multiple structural representations of source code (AST, CFG, and Data-Flow information) improve vulnerability detection compared with using a single program representation?"*

Instead of treating vulnerability detection as a simple black-box NLP problem, this project emphasizes the **compiler-analysis component**. By fusing different graphical representations, the GNN can reason about both the syntax (how the code is written), the control flow (the execution paths), and the data flow (how variables influence each other), leading to more robust and explainable vulnerability detection.

---

## 🧠 Core Graph Representations (Properties)

The project extracts and utilizes three primary graph structures from the source code:

1. **Abstract Syntax Tree (AST):**
   - **What it is:** A tree representation of the abstract syntactic structure of the source code.
   - **Role:** Captures the hierarchical syntax and grammar of the program. Useful for detecting localized syntax-level anomalies or unsafe function calls.
2. **Control Flow Graph (CFG):**
   - **What it is:** A directed graph where nodes represent basic blocks of instructions, and edges represent paths that might be traversed through the program during its execution.
   - **Role:** Captures the execution logic, loops, and conditional branches. Crucial for detecting vulnerabilities related to improper execution order, missing checks, or infinite loops.
3. **Data-Flow Graph (DFG) / Program Dependence Graph (PDG):**
   - **What it is:** A graph representing the flow of data values between different operations and variables, tracking where variables are defined and where they are used.
   - **Role:** Captures semantic dependencies. Essential for detecting vulnerabilities like use-after-free, uninitialized variables, buffer overflows, and taint analysis.

---

## ⚙️ How It Works (The Pipeline Flow)

The system is designed as a multi-stage pipeline transitioning from raw source code to a trained vulnerability classifier:

1. **Source Code Ingestion:** Raw C/C++ source code files (often from vulnerability datasets like Juliet Test Suite, SARD, or real-world open-source projects) are loaded.
2. **Compiler Analysis (Clang/LLVM):** The Clang frontend parses the C/C++ code. Using Clang Python bindings, the system traverses the parsed code to extract the AST, constructs the CFG, and performs data-flow analysis.
3. **Graph Construction & Feature Engineering:**
   - The extracted structures are converted into graph formats (nodes and edges) using `NetworkX`.
   - **Node Features:** Features are assigned to nodes (e.g., token types, variable types, operation types) using Natural Language Processing (NLP) embeddings or structural encodings.
4. **Graph Representation (PyG):** The `NetworkX` graphs are transformed into `PyTorch Geometric (PyG)` `Data` objects, ready for deep learning.
5. **Graph Neural Network (GNN):**
   - A multi-view GNN architecture (incorporating GCN, GAT, or GraphSAGE layers) processes the graphs.
   - The network aggregates information from a node's local neighborhood (syntactic, control, and data neighbors) to learn a dense embedding representing the code's semantics.
6. **Vulnerability Classification:** A pooling layer reduces the graph into a single vector, and a fully connected classifier predicts whether the code is benign or vulnerable (and potentially the type of vulnerability).
7. **Evaluation & Explainability:** The model is evaluated using standard metrics (Accuracy, Precision, Recall, F1-Score). Explainability tools can map the GNN's attention weights back to specific lines of code to highlight *why* it was flagged as vulnerable.

---

## 🛠️ Technology Stack & Languages

This project is built using a modern, research-oriented Python stack, avoiding unnecessary frameworks to maintain modularity and focus on the core compiler and ML logic.

### Languages

- **Python 3.10 / 3.11:** Primary language for extraction scripts, ML modeling, and orchestration.
- **C / C++:** Target languages being analyzed for vulnerabilities.

### Compiler Infrastructure

- **Clang & LLVM:** Used for parsing C/C++ and performing static analysis.
- **Clang Python Bindings (`libclang`):** Provides a Python interface to the Clang AST and analysis tools.

### Machine Learning & Deep Learning

- **PyTorch:** The underlying deep learning tensor library.
- **PyTorch Geometric (PyG):** A specialized library built upon PyTorch for writing and training Graph Neural Networks easily and efficiently.
- **Scikit-learn:** Used for data splitting, preprocessing, and standard evaluation metrics.

### Data & Graph Processing

- **NetworkX:** Used for the initial construction, manipulation, and visualization of the complex program graphs.
- **NumPy & Pandas:** For data manipulation, feature array processing, and dataset management.

### Visualization & Tracking

- **Matplotlib & Seaborn:** For plotting graphs, confusion matrices, and training curves.
- **Jupyter Notebooks:** For interactive data exploration, debugging ASTs, and analyzing results.

---

## 📂 Project Structure

```text
project-root/
│
├── README.md               <- Central documentation point
├── requirements.txt        <- Python dependency list
├── .gitignore              <- Git ignore rules
├── config/                 <- YAML Configuration files for pipelines and ML hyperparameters
│   └── config.yaml
│
├── data/                   <- Local data storage (ignored in version control)
│   ├── raw/                <- Original C/C++ vulnerability datasets
│   ├── interim/            <- Intermediate extracted AST/CFG representations
│   ├── processed/          <- Final PyG graph objects ready for training
│   └── splits/             <- Train/Val/Test split indices
│
├── src/                    <- Core project source code
│   ├── extraction/         <- Logic for interfacing with Clang (AST/CFG/DFG extraction)
│   ├── graphs/             <- NetworkX graph construction and merging logic
│   ├── features/           <- Node and edge feature engineering (token embeddings)
│   ├── datasets/           <- PyTorch Geometric Dataset classes and loaders
│   ├── models/             <- GNN Architectures (GCN, GAT, Multiview-GNN)
│   ├── training/           <- Training loops, loss functions, checkpointing
│   ├── evaluation/         <- Model evaluation, ablation studies, metrics
│   └── visualization/      <- Scripts for plotting graphs and training results
│
├── experiments/            <- Configuration and logs for different research experiments
│   ├── baseline_ml/        <- Baseline models (e.g., Token-based NLP)
│   ├── ast_gnn/            <- AST-only GNN experiments
│   ├── cfg_gnn/            <- CFG-only GNN experiments
│   ├── dataflow_gnn/       <- DFG-only GNN experiments
│   └── multiview_gnn/      <- Combined (AST+CFG+DFG) GNN experiments
│
├── notebooks/              <- Jupyter notebooks for research exploration
│
├── tests/                  <- Unit and integration tests for extraction and ML modules
│
└── reports/                <- Generated outputs for papers and presentations
    ├── figures/
    ├── tables/
    └── results/
```

---

## 🚀 Getting Started & Requirements

For the complete setup, pipeline, training, evaluation, visualization, and
troubleshooting instructions, see [`docs/USAGE.md`](docs/USAGE.md).
The single-command entry point is `python main.py`.

### Prerequisites

1. **Python 3.10+** installed.
2. **LLVM and Clang** installed on your system.
   - *Linux (Ubuntu):* `sudo apt-get install clang llvm libclang-dev`
   - *Windows:* Install LLVM via the official installer or `winget install LLVM`.
3. **CUDA Toolkit** (Optional, but highly recommended for GPU acceleration with PyTorch).

### Installation

1. Clone the repository:

   ```bash
   git clone <repo-url>
   cd compiler-project-sem5
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

   *(Ensure you install the correct version of PyTorch and PyG corresponding to your CUDA version from their official websites).*

### Basic Execution Flow

1. **Configure:** Set your dataset paths and model hyperparameters in `config/config.yaml`.
2. **Run:** Analyze, train, evaluate, and create reports in one command:

   ```bash
   python main.py --source tests/sample.c --label 0
   ```

3. **Advanced usage:** Use the individual Python APIs for dataset-scale
   experiments, custom splits, and research workflows.

   ```bash
   python main.py --help
   ```
