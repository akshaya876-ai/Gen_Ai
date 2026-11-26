
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# MODULE II: TRANSFORMER SELF-ATTENTION
# 4x4 MATRIX INSPECTION EDITION
# --------------------------------------------------

torch.manual_seed(42)

D_MODEL = 4


def print_matrix(name, matrix):
    """Print matrix name, shape and values."""
    print(f"\n{name}")
    print("-" * 50)

    if isinstance(matrix, torch.Tensor):
        print("Shape:", tuple(matrix.shape))
        print(matrix.detach())
    else:
        print(matrix)


def scaled_dot_product_attention(Q, K, V, causal=False):
    """
    Calculate scaled dot-product attention.

    Q = Query
    K = Key
    V = Value
    """

    d_k = Q.shape[-1]

    # Step 1: Calculate raw attention scores
    raw_scores = torch.matmul(Q, K.transpose(-2, -1))

    # Step 2: Scale scores
    scaled_scores = raw_scores / math.sqrt(d_k)

    # Step 3: Apply causal mask if enabled
    if causal:
        seq_len = Q.shape[-2]

        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool),
            diagonal=1
        )

        scaled_scores = scaled_scores.masked_fill(mask, float("-inf"))

    # Step 4: Apply softmax
    attention_weights = F.softmax(scaled_scores, dim=-1)

    # Step 5: Multiply weights with values
    output = torch.matmul(attention_weights, V)

    return raw_scores, scaled_scores, attention_weights, output


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Four 4x4 weight matrices
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X, causal=False):

        # Step 1: Generate Q, K and V
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        print_matrix("Wq", self.W_q.weight.T)
        print_matrix("Wk", self.W_k.weight.T)
        print_matrix("Wv", self.W_v.weight.T)
        print_matrix("Wo", self.W_o.weight.T)

        print_matrix("Q = X @ Wq", Q)
        print_matrix("K = X @ Wk", K)
        print_matrix("V = X @ Wv", V)

        # Step 2: Split into multiple heads
        seq_len = X.shape[0]

        Q = Q.view(seq_len, self.num_heads, self.d_k).transpose(0, 1)
        K = K.view(seq_len, self.num_heads, self.d_k).transpose(0, 1)
        V = V.view(seq_len, self.num_heads, self.d_k).transpose(0, 1)

        head_outputs = []

        # Step 3: Calculate attention for each head
        for i in range(self.num_heads):

            print("\n" + "=" * 60)
            print(f"ATTENTION HEAD {i + 1}")
            print("=" * 60)

            print_matrix(f"Head {i + 1} Q", Q[i])
            print_matrix(f"Head {i + 1} K", K[i])
            print_matrix(f"Head {i + 1} V", V[i])

            raw, scaled, weights, output = scaled_dot_product_attention(
                Q[i], K[i], V[i], causal
            )

            print_matrix("Raw Scores", raw)
            print_matrix("Scaled / Masked Scores S", scaled)
            print_matrix("Softmax Attention Weights", weights)
            print_matrix("Head Output", output)

            head_outputs.append(output)

        # Step 4: Concatenate all heads
        concatenated = torch.cat(head_outputs, dim=-1)

        print_matrix("Concatenated Head Outputs", concatenated)

        # Step 5: Final output projection
        Z = self.W_o(concatenated)

        print_matrix("Final Output Z", Z)

        return Z


def create_input(prompt):

    tokens = prompt.strip().split()

    if not tokens:
        tokens = ["Hello", "Transformer"]

    seq_len = len(tokens)

    # Create random input embeddings
    X = torch.randn(seq_len, D_MODEL)

    # Add sinusoidal positional information
    for pos in range(seq_len):
        for i in range(D_MODEL):
            if i % 2 == 0:
                X[pos, i] += math.sin(
                    pos / (10000 ** (i / D_MODEL))
                )
            else:
                X[pos, i] += math.cos(
                    pos / (10000 ** ((i - 1) / D_MODEL))
                )

    return tokens, X


def main():

    print("=" * 60)
    print("MODULE II: TRANSFORMER SELF-ATTENTION LAB")
    print("4x4 MATRIX INSPECTION EDITION")
    print("=" * 60)

    while True:

        prompt = input(
            "\nEnter a prompt (or type 'exit' to quit): "
        )

        if prompt.lower().strip() == "exit":
            print("Lab completed.")
            break

        tokens, X = create_input(prompt)

        print("\nTokens:", tokens)
        print_matrix("Input Matrix X", X)

        print("\nChoose number of attention heads:")
        print("1. Single Head")
        print("2. Two Heads")
        print("3. Four Heads")

        try:
            choice = int(input("Enter choice (1/2/4): "))

            if choice not in [1, 2, 4]:
                print("Invalid choice. Please enter 1, 2 or 4.")
                continue

        except ValueError:
            print("Please enter a valid number.")
            continue

        causal_choice = input(
            "\nEnable causal masking? (y/n): "
        ).lower().strip()

        causal = causal_choice == "y"

        print("\nCausal Masking:", causal)
        print("Number of Heads:", choice)
        print("d_model:", D_MODEL)
        print("d_k per head:", D_MODEL // choice)

        # Create and run attention model
        model = MultiHeadSelfAttention(D_MODEL, choice)

        with torch.no_grad():
            output = model(X, causal)

        print("\n" + "=" * 60)
        print("ATTENTION EXPERIMENT COMPLETED")
        print("=" * 60)


if __name__ == "__main__":
    main()
