"""Copied from jax for now.

Seems pretty well optimized based on profiling."""

import numpy as np

import jax
import jax.numpy as jnp


def _get_large_negative(dtype):
    dtype_max = jnp.finfo(dtype).max
    return jnp.asarray(-0.7 * dtype_max, dtype=dtype)


def _get_causal_mask(T, S):
    mask = jnp.tril(jnp.ones((T, S), dtype=jnp.bool_))
    return mask[None, None, :, :]


def _get_window_mask(T: int, S: int, local_window_size: tuple[int, int]):
    query_pos = jnp.array(range(T))
    key_pos = jnp.array(range(S))
    left_window, right_window = local_window_size
    left_mask = query_pos[..., None] <= key_pos[..., None, :] + left_window
    right_mask = query_pos[..., None] >= key_pos[..., None, :] - right_window
    return jnp.logical_and(right_mask, left_mask)[None, None, :, :]


def _get_padding_mask_logits(T, S, q_seqlen, kv_seqlen):
    q_mask = True
    kv_mask = True
    if q_seqlen is not None:
        q_indices = jnp.arange(0, T)[None, :, None]
        q_mask = q_indices < q_seqlen[:, None, None]
    if kv_seqlen is not None:
        kv_indices = jnp.arange(0, S)[None, None, :]
        kv_mask = kv_indices < kv_seqlen[:, None, None]
    mask = jnp.logical_and(q_mask, kv_mask)
    return mask[:, None, :, :]


def _get_padding_mask_encoded(T, q_seqlen):
    q_indices = jnp.arange(0, T)[None, :]
    mask = q_indices < q_seqlen[:, None]
    return mask[:, :, None, None]


def _apply_masks(logits, mask, is_causal, q_seqlen, kv_seqlen, local_window_size):
    if mask is None and not is_causal and q_seqlen is None and kv_seqlen is None:
        return logits

    combined_mask = jnp.ones_like(logits, dtype=jnp.bool_)
    if mask is not None:
        assert mask.dtype == jnp.bool_
        combined_mask = jnp.logical_and(combined_mask, mask)

    T, S = logits.shape[2], logits.shape[3]

    if is_causal:
        mask = _get_causal_mask(T, S)
        combined_mask = jnp.logical_and(combined_mask, mask)

    if local_window_size is not None:
        mask = _get_window_mask(T, S, local_window_size)
        combined_mask = jnp.logical_and(combined_mask, mask)

    if q_seqlen is not None or kv_seqlen is not None:
        mask = _get_padding_mask_logits(T, S, q_seqlen, kv_seqlen)
        combined_mask = jnp.logical_and(combined_mask, mask)

    large_negative_number = _get_large_negative(logits.dtype)
    padded_logits = jnp.where(combined_mask, logits, large_negative_number)
    return padded_logits


def _dot_product_attention_core(
    query,
    key,
    value,
    bias,
    mask,
    is_causal,
    scale,
    q_seqlen,
    kv_seqlen,
    local_window_size,
):
    logits = jnp.einsum("BTNH,BSNH->BNTS", query, key)

    logits *= jnp.array(scale, dtype=logits.dtype)

    if bias is not None:
        logits += bias.astype(logits.dtype)

    padded_logits = _apply_masks(
        logits, mask, is_causal, q_seqlen, kv_seqlen, local_window_size
    )

    # Softmax and it is always carried out in fp32.
    probs = jax.nn.softmax(padded_logits.astype(jnp.float32), axis=-1).astype(
        padded_logits.dtype
    )

    encoded = jnp.einsum("BNTS,BSNH->BTNH", probs, value)
    if q_seqlen is not None and kv_seqlen is not None:
        mask = _get_padding_mask_encoded(encoded.shape[1], q_seqlen)
        encoded *= mask.astype(encoded.dtype)
    return encoded


def _dot_product_attention_xla(
    query,
    key,
    value,
    bias,
    mask,
    is_causal: bool,
    scale: float,
    q_seqlen,
    kv_seqlen,
    local_window_size: tuple[int, int] | None,
):

    B, T, N, H = query.shape
    _, S, K, _ = key.shape
    G = N // K

    query = jnp.reshape(query, (B, T, K, G, H))

    def _reshape_to_grouped(t):
        if t is not None:
            tB, tN, tT, tS = t.shape
            if tN == 1:
                t = jnp.broadcast_to(t[:, :, None, :, :], (tB, tN, G, tT, tS))
            else:
                assert tN == N
                t = jnp.reshape(t, (tB, K, G, tT, tS))
        return t

    bias = _reshape_to_grouped(bias)
    mask = _reshape_to_grouped(mask)
    vmapped_fn = jax.vmap(
        _dot_product_attention_core,
        in_axes=(3, None, None, 2, 2, None, None, None, None, None),
        out_axes=3,
    )
    encoded = vmapped_fn(
        query,
        key,
        value,
        bias,
        mask,
        is_causal,
        scale,
        q_seqlen,
        kv_seqlen,
        local_window_size,
    )
    encoded = jnp.reshape(encoded, (B, T, N, H))
    return encoded


def dot_product_attention(
    query,
    key,
    value,
    bias=None,
    mask=None,
    *,
    scale: float | None = None,
    is_causal: bool = False,
    query_seq_lengths=None,
    key_value_seq_lengths=None,
    local_window_size: int | tuple[int, int] | None = None,
    implementation=None,
):
    r"""Scaled dot product attention function.

    Computes the attention function on Query, Key, and Value tensors:

    .. math::

      \mathrm{Attention}(Q, K, V)=\mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V

    If we define :code:`logits` as the output of :math:`QK^T` and the
    :code:`probs` as the output of :math:`softmax`.

    Throughout this function, we utilize the following uppercase letters to
    represent the shape of array::

      B = batch size
      S = length of the key/value (source)
      T = length of the query (target)
      N = number of attention heads
      H = dimensions of each attention head
      K = number of key/value heads
      G = number of groups, which equals to N // K

    Args:
      query: query array; shape :code:`(BTNH|TNH)`
      key: key array: shape :code:`(BSKH|SKH)`. When `K` equals `N`, multi-headed
        attention (MHA https://arxiv.org/abs/1706.03762) is performed. Otherwise,
        grouped query attention (GQA https://arxiv.org/abs/2305.13245) is
        performed if `N` is a multiple of `K`, and multi-query attention (MQA
        https://arxiv.org/abs/1911.02150) is performed if `K == 1` (a special case
        of GQA).
      value: value array, should have the same shape as the `key` array.
      bias: optional, bias array to be added to logits; The shape must be 4D and
        be broadcastable to :code:`(BNTS|NTS)`.
      mask: optional, mask array used to filter out logits. It is a boolean mask
        where `True` indicates the element should take part in attention. For an
        additive mask, users should pass it to `bias`. The shape must be 4D and be
        broadcastable to :code:`(BNTS|NTS)`.
      scale: scale for the logits. If None, the scale will be set to 1 divided by
        the square root of query's head dimension (i.e. H).
      is_causal: If true, causal attention will be applied. Note, some
        implementations like `xla` will generate a mask tensor and apply it to the
        logits to mask out the non-causal parts of the attention matrix, but other
        implementations like `cudnn` will avoid computing the non-causal regions,
        providing speedups.
      query_seq_lengths: `int32` array of sequence lengths for query; shape
        :code:`(B)`
      key_value_seq_lengths: `int32` array of sequence lengths for key and value;
        shape :code:`(B)`
      local_window_size: Window sizes to make self attention to attend to each
        token's local window. If set, this specifies the (left_window_size,
        right_window_size) for each token. E.g., if local_window_size == (3, 2)
        and the sequence is [0, 1, 2, 3, 4, 5, c, 7, 8, 9], token `c` can attend
        to [3, 4, 5, c, 7, 8]. If a single int is given, it will be intepreted as
        a symmetric window (window_size, window_size).
      implementation: A string to control which implementation backend to use.
        Supported strings are `xla`, `cudnn` (cuDNN flash attention). It defaults
        to `None`, which will automatically select the best available backend.
        Note, `cudnn` supports only a subset of shapes/dtypes, and an exception
        will be thrown if its not supported.

    Returns:
      An array of the attention output with the same shape as :code:`query`.
    """
    output_shape = jnp.asarray(query).shape

    def _ensure_4d(t):
        t = jnp.asarray(t)
        dims_to_add = 4 - t.ndim
        if dims_to_add > 0:
            return jnp.expand_dims(t, axis=tuple(range(dims_to_add)))
        return t

    query_arr = _ensure_4d(query)
    key_arr = _ensure_4d(key)
    value_arr = _ensure_4d(value)
    bias = _ensure_4d(bias) if bias is not None else None
    mask = _ensure_4d(mask) if mask is not None else None
    if query_seq_lengths is not None:
        query_seq_lengths = jnp.asarray(query_seq_lengths)
    if key_value_seq_lengths is not None:
        key_value_seq_lengths = jnp.asarray(key_value_seq_lengths)
    if isinstance(local_window_size, int):
        local_window_size = (local_window_size, local_window_size)

    def _check_shape_and_dtype(t, shape, dtype, name: str) -> None:
        if t is None:
            return
        if t.ndim != len(shape):
            raise ValueError(f"{name} ndim should be {len(shape)}, but got {t.ndim}")
        if dtype is not None and t.dtype != dtype:
            raise ValueError(f"{name} dtype should be {dtype}, but got {t.dtype}")
        for i in range(t.ndim):
            if shape[i] != -1 and t.shape[i] != shape[i]:
                raise ValueError(f"{name} shape should be {shape}: but got {t.shape}")

    B, S, K, H = key_arr.shape
    _check_shape_and_dtype(value_arr, [B, S, K, H], key_arr.dtype, "value")
    _check_shape_and_dtype(query_arr, [B, -1, -1, H], key_arr.dtype, "query")
    _check_shape_and_dtype(mask, [-1] * 4, jnp.bool_, "mask")
    _check_shape_and_dtype(bias, [-1] * 4, None, "bias")
    _check_shape_and_dtype(query_seq_lengths, [B], jnp.int32, "query_seq_lengths")
    _check_shape_and_dtype(
        key_value_seq_lengths, [B], jnp.int32, "key_value_seq_lengths"
    )
    if query_arr.shape[-2] % K != 0:
        raise ValueError(
            f"The number of query heads must be a multiple of "
            f"key/value heads, but got {query_arr.shape[-2]} vs {K}"
        )

    scale_val = (1.0 / np.sqrt(H)) if scale is None else scale

    out = _dot_product_attention_xla(
        query_arr,
        key_arr,
        value_arr,
        bias,
        mask,
        is_causal=is_causal,
        scale=scale_val,
        q_seqlen=query_seq_lengths,
        kv_seqlen=key_value_seq_lengths,
        local_window_size=local_window_size,
    )

    return jnp.reshape(out, output_shape)
