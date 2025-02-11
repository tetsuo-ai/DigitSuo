# DigitSuo

## $TETSUO on Solana

**Contract Address**: `8i51XNNpGaKaj4G4nDdmQh95v4FKAxw8mhtaRoKd9tE8`

[![Twitter](https://img.shields.io/badge/Twitter-Follow%20%407etsuo-1DA1F2)](https://x.com/7etsuo)
[![Discord](https://img.shields.io/badge/Discord-Join%20Our%20Community-7289DA)](https://discord.gg/tetsuo-ai)

---

A neural network–based digit recognition project using the MNIST dataset.

<img src="https://github.com/user-attachments/assets/544f7d99-563e-4766-af2b-db1649b23d23" alt="image" width="500">

## Installation

1. **Clone the Repository**
   ```bash
   git clone git@github.com:tetsuo-ai/DigitSuo.git
   cd DigitSuo
   ```

2. **Install Dependencies**

   **Ubuntu/Debian:**
   ```bash
   sudo apt-get install gcc libncurses-dev zlib1g-dev doxygen graphviz
   ```

   **Fedora/RHEL:**
   ```bash
   sudo dnf install gcc ncurses-devel zlib-devel doxygen graphviz
   ```

## Building the Project

Use the provided Makefile for the following targets:

- **Recognition Interface:**
  ```bash
  make
  ```
  Compiles source files in `src/` into the executable `digit_recognition`.

- **Training Program:**
  ```bash
  make train
  ```
  Compiles `train.c` with OpenMP support into the executable `train`.

- **Documentation:**
  ```bash
  make docs
  ```
  Generates HTML documentation in `docs/html/` using Doxygen.

- **Clean Build Artifacts:**
  ```bash
  make clean
  ```

- **Delete Debug Log:**
  ```bash
  make clean_debug
  ```

- **Delete Debug Log:**
  ```bash
  make clean_docs
  ```

## Usage

### Training the Network

Run the training program:
```bash
./train
```
This will:
- Load and preprocess the MNIST dataset (files: `train-images-idx3-ubyte.gz` and `train-labels-idx1-ubyte.gz`).
- Apply data augmentation.
- Train the neural network.
- Save optimized weights to `src/weights.h`.

### Using the Recognition Interface

Run the recognition interface:
```bash
./digit_recognition
```

#### Controls
- **Mouse/Arrow Keys**: Draw digits
- **Number Keys (0-9)**: Load example digits
- **Enter**: Submit for recognition
- **C**: Clear drawing
- **Q**: Quit application

## Technical Details

### Neural Network Architecture

```plaintext
Input Layer (784 neurons)
    │
    │   Dense connections with ReLU activation
    │   Weights initialized using He initialization
    ▼
Hidden Layer (256 neurons)
    │
    │   Dense connections with Softmax activation
    │   Xavier/Glorot initialization
    ▼
Output Layer (10 neurons)
```

### Training Pipeline

1. **Data Preprocessing**
   - Load the MNIST dataset with zlib decompression.
   - Apply data augmentation:
     - Random rotation (±10°)
     - Random shifts (±5 pixels)
     - Gaussian blur (σ = 0.3)
   - Generate balanced mini-batches.

2. **Optimization**
   - Mini-batch gradient descent with momentum.
   - Learning rate decay schedule.
   - Early stopping with patience.
   - OpenMP for parallel processing.

### Performance Metrics

- Training accuracy: >98% on the MNIST test set.
- Inference latency: <10ms per prediction.
- Memory footprint: ~2MB during inference.
- Training time: ~5 minutes on a modern CPU.

## Documentation

### Generating Documentation

Generate the documentation using:
```bash
make docs
```

### Viewing Documentation

Start the documentation server:
```bash
python3 serve_docs.py
```
Access the documentation at [http://localhost:8000](http://localhost:8000).

## Project Structure

```plaintext
.
├── Doxyfile                  # Doxygen configuration file
├── doxygen-custom.css        # Custom CSS for documentation
├── Makefile                  # Build configuration
├── README.md                 # Project documentation
├── docs/                     # Documentation directory (generated)
├── serve_docs.py             # Script to serve documentation
├── train-images-idx3-ubyte.gz # MNIST training images (compressed)
├── train-labels-idx1-ubyte.gz # MNIST training labels (compressed)
├── train.c                   # Training program source
└── src/                      # Source code for recognition interface
    ├── main.c
    ├── draw_interface.c
    ├── neural_net.c
    ├── neural_net.h
    ├── draw_interface.h
    ├── utils.c
    ├── utils.h
    └── weights.h
```

## Customization

Adjust neural network parameters in `train.c`:
```c
#define HIDDEN_SIZE 256      // Number of hidden neurons
#define BATCH_SIZE 64        // Training batch size
#define NUM_EPOCHS 10        // Maximum training epochs
#define BASE_LR 0.1f         // Initial learning rate
```
