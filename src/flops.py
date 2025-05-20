import numpy as np

def rmsnorm_flops(B, L, d, per_vector_constant=10):
    """
    Compute the FLOPs for a single RMSNorm pass.
    By default, assume each token vector (length d) over (B,L,d) requires (4*d + per_vector_constant) FLOPs.

    - Where: 2*d from squaring + summation, 2*d from division + multiplication,
      plus sqrt and other fixed-cost operations ~ per_vector_constant (default=10)
    """
    return (B * L) * (4 * d + per_vector_constant)


def attn_flops(B, L, d):
    """
    Standard multi-head self-attention FLOPs (ignoring head splits):
      => 8*B*L*d^2  + 4*B*L^2*d
    """
    return 8 * B * L * (d**2) + 4 * B * (L**2) * d


def mlp_flops(B, L, d, d_ff, act_extra=11):
    """
    MLP/FFN main matrix multiplication + activation/gating overhead.
    Common simplification: 4 * B*L*d*d_ff  +  (SwiGLU-style activation ~ B*L*d_ff*constant)
    The act_extra parameter represents per-element activation overhead.
    """
    # Linear FC layers (gate_proj, up_proj, down_proj): simplified as 6 * B*L*d*d_ff
    fc_flops = 6 * B * L * d * d_ff

    # Assume activation functions (e.g., SiLU) + gating cost ~ act_extra FLOPs/element in d_ff
    # e.g., 10–12 FLOPs/element depending on implementation
    act_flops = act_extra * B * L * d_ff

    return fc_flops + act_flops


def qwen_forward_flops_v2(
    B: int,         # batch size
    L: int,         # seq length
    N: int,         # number of layers (24 for Qwen-0.5B)
    d: int,         # hidden_size=896
    d_ff: int,      # intermediate_size=4864
    V: int,         # vocab_size=151936
    # Parameters below can be adjusted based on your breakdown
    rmsnorm_const: int = 11,  # Per-vector constant FLOPs in RMSNorm, default=10 (e.g., sqrt/log)
    act_extra: int = 14,      # Extra FLOPs per element for activation/gating in MLP
    include_pos_add: bool = True
):
    """
    Compute the forward FLOPs of a Qwen-style Transformer using a breakdown-based formula.

    FLOPs_forward = 
        [optional positional addition] + 
        sum_{l=1..N} [ 2*RMSNorm + Attention + MLP ] +
        [ final RMSNorm ] +
        [ lm_head (output projection) ].
    """

    # 1) Optional: element-wise addition with positional encodings => B*L*d additions
    flops_positional = B * L * d if include_pos_add else 0

    # 2) Each layer: (2 x RMSNorm) + Attention + MLP
    layer_flops = 0

    # 2a) RMSNorm (twice):
    #     = 2 * rmsnorm_flops(B, L, d, rmsnorm_const)
    flops_rms_2 = 2 * rmsnorm_flops(B, L, d, rmsnorm_const)

    # 2b) Attention:
    flops_attn = attn_flops(B, L, d)

    # 2c) MLP:
    flops_mlp = mlp_flops(B, L, d, d_ff, act_extra)

    # Total for one layer:
    flops_layer = flops_rms_2 + flops_attn + flops_mlp

    # N layers:
    flops_all_layers = N * flops_layer

    # 3) Final RMSNorm:
    flops_final_norm = rmsnorm_flops(B, L, d, rmsnorm_const)

    # 4) lm_head projection: 
    #    => (B*L, d) -> (B*L, V), ~ 2*B*L*d*V
    flops_lm_head = 2 * B * L * d * V

    # Total:
    flops_forward = (
        flops_positional 
        + flops_all_layers
        + flops_final_norm
        + flops_lm_head
    )

    return flops_forward

def qwen_forward_and_backward_flops_v2(
    B: int,         # batch size
    L: int,         # seq length
    N: int,         # number of layers (24 for Qwen-0.5B)
    d: int,         # hidden_size=896
    d_ff: int,      # intermediate_size=4864
    V: int,         # vocab_size=151936
    rmsnorm_const: int = 11,  # Per-vector constant FLOPs in RMSNorm, default=10 (e.g., sqrt/log)
    act_extra: int = 14,      # Extra FLOPs per element for activation/gating in MLP
    include_pos_add: bool = True
):
    """
    Compute the total FLOPs of Qwen's forward + backward pass.
    Assumes backward = 2 × forward.
    """
    fwd = qwen_forward_flops_v2(
        B=B,
        L=L,
        N=N,
        d=d,
        d_ff=d_ff,
        V=V,
        rmsnorm_const=rmsnorm_const,
        act_extra=act_extra,
        include_pos_add=include_pos_add
    )
    return 3 * fwd  # (Forward + Backward)

def lora_added_flops_one_layer(
    batch_size: int,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    rank: int
) -> float:
    """
    Compute the additional FLOPs introduced by the LoRA branch (A and B matrices)
    for a single Linear(in_dim, out_dim) layer in one forward + backward training step.

    Parameters
    ----------
    batch_size : int
        Batch size (B).
    seq_len : int
        Sequence length (L).
    in_dim : int
        Input dimension of the original Linear layer.
    out_dim : int
        Output dimension of the original Linear layer.
    rank : int
        LoRA rank (r).

    Returns
    -------
    float
        Additional FLOPs from LoRA branch (forward + backward).

    Notes
    -----
    - The original linear weights are frozen, so their gradients are not computed.
    - Only the LoRA branch (A, B) is trainable, so we count their forward + backward FLOPs.
    - Forward FLOPs for LoRA ≈ 2 * B * L * r * (in_dim + out_dim)
    - Training FLOPs (forward + backward) ≈ 3 * LoRA forward
    """
    BL = batch_size * seq_len
    forward_flops_lora = 2.0 * BL * rank * (in_dim + out_dim)
    return forward_flops_lora


def lora_added_flops_forward(
    batch_size: int,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    rank: int,
    num_layers: int
) -> float:
    """
    Compute the additional LoRA FLOPs (only forward pass) across multiple layers.

    Parameters
    ----------
    batch_size : int
        Batch size (B).
    seq_len : int
        Sequence length (L).
    in_dim : int
        Input dimension of the Linear layer.
    out_dim : int
        Output dimension of the Linear layer.
    rank : int
        LoRA rank (r).
    num_layers : int
        Number of layers with LoRA-inserted linear projections.

    Returns
    -------
    float
        Total added FLOPs from LoRA branches (forward only).
    """
    return num_layers * lora_added_flops_one_layer(
        batch_size=batch_size,
        seq_len=seq_len,
        in_dim=in_dim,
        out_dim=out_dim,
        rank=rank
    )


def lora_added_flops(
    batch_size: int,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    rank: int,
    num_layers: int
) -> float:
    """
    Compute the total additional FLOPs introduced by LoRA branches during training
    (forward + backward) across multiple layers.

    Parameters
    ----------
    batch_size : int
        Batch size (B).
    seq_len : int
        Sequence length (L).
    in_dim : int
        Input dimension of the Linear layer.
    out_dim : int
        Output dimension of the Linear layer.
    rank : int
        LoRA rank (r).
    num_layers : int
        Number of LoRA-inserted layers.

    Returns
    -------
    float
        Total additional FLOPs for one training step (forward + backward).
    """
    return 3 * lora_added_flops_forward(
        batch_size=batch_size,
        seq_len=seq_len,
        in_dim=in_dim,
        out_dim=out_dim,
        rank=rank,
        num_layers=num_layers
    )

if __name__ == "__main__":

    ###################################
    # Example for Qwen2.5-0.5B:
    ####################################
    print("\n====== Qwen2.5-0.5B FLOPs ======")
    B, L = 4, 512
    N = 24
    d = 896
    d_ff = 4864
    V = 151936

    total_flops = qwen_forward_flops_v2(B, L, N, d, d_ff, V)
    print(f"Qwen forward pass FLOPs = {total_flops:.3e}, {np.log10(total_flops):.3f}")
    print(f"Qwen training FLOPs = {3*total_flops:.3e}, {np.log10(3*total_flops):.3f}")

    ####################################
    # Example for LoRA:
    ####################################
    print("\n====== LoRA FLOPs ======")
    in_dim = 896
    out_dim = 896
    rank = 4
    num_layers = 24

    q_lora = lora_added_flops(B, L, 896, 896, rank, num_layers)
    print(f"[LoRA@q_proj] => {q_lora:.3e} FLOPs (one step)")

    # 2) v_proj => (in_dim=896, out_dim=128)
    v_lora = lora_added_flops(B, L, 896, 128, rank, num_layers)
    print(f"[LoRA@v_proj] => {v_lora:.3e} FLOPs (one step)")

    # If LoRA is applied to both q_proj and v_proj in the same layer:
    add_lora = q_lora + v_lora
    print(f"[LoRA@one layer q,v] => {add_lora:.3e} FLOPs (one step)")


    #######################################
    ### For our whole training experiments:
    #######################################
    print("\n====== FLOPs for all experiments ======")
    context_lengths = [512, 512, 512, 512, 512, 128, 512, 768, 512]
    number_of_steps = [2000, 2000, 2000, 2000, 2000, 1000, 1000, 1000, 2000]
    lora_rank = [4, 4, 4, 2, 8, 8, 8, 8, 8]

    d_in = 896
    d_out_q = 896
    d_out_v = 128

    total_flops_for_training_all = 0
    for i in range(len(context_lengths)):
        L = context_lengths[i]
        num_steps = number_of_steps[i]

        total_flops_for_training = qwen_forward_and_backward_flops_v2(
            B,
            L,
            N,
            d,
            d_ff,
            V,
        ) * num_steps
        print(f"Experiment {i+1}: Training Qwen for {num_steps} steps => total FLOPs = {np.log10(total_flops_for_training):.6f}")

        # FLOPs for LoRA branches:
        rank = lora_rank[i]

        # 1) q_proj => (in_dim=896, out_dim=896)
        q_lora = lora_added_flops(
            B,
            L,
            d_in,
            d_out_q,
            rank,
            N
        ) * num_steps
        
        # 2) v_proj => (in_dim=896, out_dim=128)
        v_lora = lora_added_flops(
            B,
            L,
            d_in,
            d_out_v,
            rank,
            N
        ) * num_steps

        lora_flops = q_lora + v_lora
        print(f"Experiment {i+1}: LoRA added FLOPs for {num_steps} steps = {np.log10(lora_flops):.6f}")

        exp_total_flops = total_flops_for_training + lora_flops
        print(f"Experiment {i+1}: Total Training FLOPs for {num_steps} steps = {np.log10(exp_total_flops):.6f}")

        # Accumulate total FLOPs for all experiments
        total_flops_for_training_all += exp_total_flops
        
    print(f"log10 of total FLOPs = {np.log10(total_flops_for_training_all):.6f}")