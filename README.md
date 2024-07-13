# Attention Is All You Need

This repo is a work-in-progress towards the goal of a minimal implementation of [transformers with multi-head self attention](https://arxiv.org/abs/1706.03762), for my own curiosity and deeper understanding. The model is trained and evaluated on a toy dataset where the task is to reverse a sequence of integers.  

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/naivoder/AttentionIsAllYouNeed.git
    cd AttentionIsAllYouNeed
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the training script:

    ```bash
    python main.py
    ```

## Acknowledgements

Special thanks to [Aladdin Persson](https://www.youtube.com/@AladdinPersson) for his explanation of torch.einsum for the attention mechanism.
