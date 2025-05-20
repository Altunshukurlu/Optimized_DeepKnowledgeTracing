# Optimized Deep Knowledge Tracing

A deep learning-based knowledge tracing system that models student learning over time. This project implements knowledge tracing models to predict student performance and track learning progress. Unlike the original implementation, which primarily relied on stochastic gradient descent (SGD), our approach explores modern optimization algorithms such as AdamW and Adagrad using PyTorch, aiming to improve convergence and generalization performance.

## Original Paper

This project is based on the paper "Deep Knowledge Tracing" by Chris Piech, Jonathan Spencer, Jonathan Huang, Surya Ganguli, Mehran Sahami, Leonidas Guibas, and Jascha Sohl-Dickstein. The paper was published at NIPS 2015.

[Deep Knowledge Tracing Paper](http://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf)

Here is my paper [Improving Deep Knowledge Tracing via Gated Architectures and Adaptive Optimization](Paper: https://arxiv.org/abs/2504.20070)

## Project Structure

```
.../
├── app/
│   └── main.py          # Main application code
├── data/
│   ├── assistments/     # Assistments dataset
│   └── synthetic/       # Synthetic dataset
└── requirements.txt     # Python dependencies
```

## Features

- Deep Knowledge Tracing implementation
- Support for multiple datasets (Assistments and synthetic)
- Model training and evaluation
- Student performance prediction
- Learning progress tracking

## Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/altunshukurlu/Optimized_DeepKnowledgeTracing.git
cd Optimized_DeepKnowledgeTracing
```

2. Create and activate a virtual environment:

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python -c "import torch, pandas; print(torch.__version__, pandas.__version__)"
```

## Usage

1. Prepare your dataset in the appropriate format and place it in the `data` directory
2. Change the optimization in RNN._init_ (current set to AdamW)
3. Run the main application:
```bash
python app/main.py
```

## Data Organization

The project supports two types of datasets:

- **Assistments Dataset**: Real-world educational data
- **Synthetic Dataset**: Generated data for testing and validation

Place your datasets in the respective directories under the `data` folder.

## FAQ

### Q: How do you handle multiple students during training?
A: Sequences of different lengths (from different students) are padded to the same length for efficient training.

### Q: Does the training code have a termination condition?
A: No. The model is saved after each epoch, allowing you to:
- Run training until you decide to stop
- Resume training from any saved checkpoint
- Start training from any previously saved model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

MIT License

Copyright (c) 2024 Altun Shukurlu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so.
