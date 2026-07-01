import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttentionWithKVCache(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.head_size = head_size
        self.q_proj = nn.Linear(n_embd, head_size, bias=False)
        self.k_proj = nn.Linear(n_embd, head_size, bias=False)
        self.v_proj = nn.Linear(n_embd, head_size, bias=False)

    def prefill(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_size)

        seq_len = x.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ v

        cache = {"k": k, "v": v}
        return out, cache

    def decode_one_token(self, x_new, cache):
        q_new = self.q_proj(x_new)
        k_new = self.k_proj(x_new)
        v_new = self.v_proj(x_new)

        k_all = torch.cat([cache["k"], k_new], dim=1)
        v_all = torch.cat([cache["v"], v_new], dim=1)

        scores = q_new @ k_all.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_size)
        weights = F.softmax(scores, dim=-1)
        out = weights @ v_all

        new_cache = {"k": k_all, "v": v_all}
        return out, new_cache, q_new, k_new, v_new, k_all, v_all, weights


def main():
    torch.manual_seed(42)

    batch_size = 1
    prompt_len = 4
    n_embd = 6
    head_size = 3

    attn = SingleHeadAttentionWithKVCache(n_embd=n_embd, head_size=head_size)

    x_prompt = torch.randn(batch_size, prompt_len, n_embd)
    out_prefill, cache = attn.prefill(x_prompt)

    print("Prefill:")
    print("x_prompt.shape:", x_prompt.shape)
    print("out_prefill.shape:", out_prefill.shape)
    print("cache['k'].shape:", cache["k"].shape)
    print("cache['v'].shape:", cache["v"].shape)

    x_new = torch.randn(batch_size, 1, n_embd)
    out_decode, new_cache, q_new, k_new, v_new, k_all, v_all, weights = attn.decode_one_token(x_new, cache)

    print("\nDecode one token:")
    print("x_new.shape:", x_new.shape)
    print("q_new.shape:", q_new.shape)
    print("k_new.shape:", k_new.shape)
    print("v_new.shape:", v_new.shape)

    print("\nAfter appending new K/V to cache:")
    print("k_all.shape:", k_all.shape)
    print("v_all.shape:", v_all.shape)
    print("new_cache['k'].shape:", new_cache["k"].shape)
    print("new_cache['v'].shape:", new_cache["v"].shape)

    print("\nDecode attention:")
    print("weights.shape:", weights.shape)
    print("out_decode.shape:", out_decode.shape)
    print("weights sum:", weights.sum(dim=-1))

    print("\nMeaning:")
    print("Prefill computes and caches K/V for the whole prompt.")
    print("Decode computes Q/K/V only for the new token.")
    print("The new token attends to all cached K/V plus its own new K/V.")
    print("KV cache avoids recomputing K/V for old tokens at every decode step.")


if __name__ == "__main__":
    main()
