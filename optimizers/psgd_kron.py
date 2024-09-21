from typing import Any, List, Optional, Union, Callable, Tuple
from functools import partial

import jax
from jax import vmap
import jax.numpy as jnp
import opt_einsum
from optax import tree_utils as otu
from optax._src import base, transform
from optax._src.linear_algebra import global_norm
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype
from optax._src.combine import chain


def scale_by_kron(
    preconditioner_update_probability: Union[float, Callable[[int], float]] = 0.5,
    b1: float = 0.9,
    nesterov: bool = False,
    max_size_triangular: int = 4096,
    max_skew_triangular: int = 10,
    precond_lr: Union[float, Callable[[int], float]] = 0.1,
    precond_init_scale: float = 0.1,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precision: str = "bfloat16",
    scanned_layers: Optional[base.Params] = None,
) -> base.GradientTransformationExtraArgs:
    """
    Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    Args:
        preconditioner_update_probability: float, probability of updating the
            preconditioner.
        b1: float, momentum parameter.
        nesterov: bool, whether to use Nesterov momentum.
        max_size_triangular: int, max size for preconditioner to be triangular.
        max_skew_triangular: int, max skew for preconditioner to be triangular.
        precond_lr: float or callable, learning rate for the preconditioner.
        precond_init_scale: float, initial scale for the preconditioner.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum accumulator.
            Defaults to the same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioner.
        precision: str, precision for matmul, 'bfloat16', 'tensorfloat32', 'float32'.
        scanned_layers: optional base.Params, tree of bool indicating scanned layers.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    mu_dtype = canonicalize_dtype(mu_dtype)
    precond_dtype = canonicalize_dtype(precond_dtype)

    def init_fn(params):
        if scanned_layers is None:
            scanned_layers_ = jax.tree.map(lambda _: False, params)
        else:
            scanned_layers_ = scanned_layers

        # momentum
        mu = None
        if b1 > 0:
            mu = otu.tree_zeros_like(params, mu_dtype)

        # preconditioners
        Qs = [
            _init_Q_exprs(
                t[0] if s else t,
                precond_init_scale,
                max_size_triangular,
                max_skew_triangular,
                precond_dtype,
            )[0]
            for t, s in zip(jax.tree.leaves(params), jax.tree.leaves(scanned_layers_))
        ]
        # broadcast for scanned layers
        Qs = [
            (
                jax.tree.map(
                    lambda d: jnp.repeat(jnp.expand_dims(d, 0), t.shape[0], axis=0), q
                )
                if s
                else q
            )
            for q, t, s in zip(
                Qs, jax.tree.leaves(params), jax.tree.leaves(scanned_layers_)
            )
        ]
        Qs = jax.tree.structure(params).unflatten(Qs)

        # Calculate sizes for nu (preconditioner) and mu (momentum)
        Qs_n_elements = sum([q.size for q in jax.tree.leaves(Qs)])
        Qs_size_MB = sum(
            [q.size * q.dtype.itemsize / (2**20) for q in jax.tree.leaves(Qs)]
        )
        if jax.process_index() == 0:
            print(
                f"PSGD Preconditioners size: {Qs_n_elements} elements, "
                f"{Qs_size_MB:.2f} MB"
            )
        if mu is not None:
            mu_n_elements = sum([p.size for p in jax.tree.leaves(mu)])
            mu_size_MB = sum(
                [p.size * p.dtype.itemsize / (2**20) for p in jax.tree.leaves(mu)]
            )
            if jax.process_index() == 0:
                print(
                    f"PSGD Momentum size: {mu_n_elements} elements, {mu_size_MB:.2f} MB"
                )

        # initial state
        return dict(
            count=jnp.zeros([], jnp.int32),
            key=jax.random.PRNGKey(5318008),
            mu=mu,
            Qs_preconditioners=Qs,
        )

    def update_fn(updates: base.Updates, state: dict, params: base.Params = None):
        del params
        count_inc = safe_int32_increment(state["count"])
        key = state["key"]

        if scanned_layers is None:
            scanned_layers_ = jax.tree.map(lambda _: False, updates)
        else:
            scanned_layers_ = scanned_layers

        update_prob_in = preconditioner_update_probability
        if isinstance(preconditioner_update_probability, Callable):
            update_prob_in = preconditioner_update_probability(count_inc)

        precond_lr_in = precond_lr
        if isinstance(precond_lr, Callable):
            precond_lr_in = precond_lr(count_inc)

        # momentum
        mu = None
        if state["mu"] is not None:
            updates, mu = _apply_momentum(updates, state["mu"], count_inc, b1, nesterov)

        # flatten pytrees
        updates, grads_structure = jax.tree.flatten(updates)
        Qs = grads_structure.flatten_up_to(state["Qs_preconditioners"])
        flat_scanned_layers = grads_structure.flatten_up_to(scanned_layers_)

        # get einsum expressions
        expressions = [
            _init_Q_exprs(
                t[0] if s else t,
                precond_init_scale,
                max_size_triangular,
                max_skew_triangular,
                precond_dtype,
                existing_Q=jax.tree.map(lambda d: d[0], Q) if s else Q,
            )
            for t, s, Q in zip(updates, flat_scanned_layers, Qs)
        ]

        # update preconditioner
        key, subkey = jax.random.split(key)
        do_update = jax.random.uniform(subkey, dtype=jnp.float32) < update_prob_in

        integrate_out_v = False
        if integrate_out_v:
            Vs = [None for _ in range(len(updates))]
        else:
            key, subkey = jax.random.split(key)
            Vs_keys = jax.random.split(subkey, len(updates))
            Vs = [
                jax.random.normal(k, shape=g.shape, dtype=g.dtype)
                for k, g in zip(Vs_keys, updates)
            ]

        update_precond_fn = partial(
            _update_precond_kron_math, precond_lr=precond_lr_in, precision=precision
        )

        def update_preconditioner():
            new_Qs = []
            for Q, g, v, expr, s in zip(
                Qs, updates, Vs, expressions, flat_scanned_layers
            ):
                if s:
                    in_axes = (0, 0, None, None) if v is None else (0, 0, 0, None)
                    new_Qs.append(
                        vmap(update_precond_fn, in_axes=in_axes)(Q, g, v, expr)
                    )
                else:
                    new_Qs.append(update_precond_fn(Q, g, v, expr))
            new_Qs = otu.tree_cast(new_Qs, precond_dtype)
            return new_Qs

        Qs = jax.lax.cond(do_update, update_preconditioner, lambda: Qs)

        # balance preconditioners about every 100 updates
        def balance_Q(Q: List[jax.Array]):
            norms = jnp.array([jnp.max(jnp.abs(q)) for q in Q], dtype=jnp.float32)

            large_idx, small_idx = jnp.argmax(norms), jnp.argmin(norms)
            large = jax.lax.dynamic_index_in_dim(norms, large_idx, keepdims=False)
            small = jax.lax.dynamic_index_in_dim(norms, small_idx, keepdims=False)

            rho = jnp.sqrt(large / small)

            to_mul = jnp.ones_like(norms)
            to_mul = jax.lax.dynamic_update_index_in_dim(to_mul, 1 / rho, large_idx, 0)
            to_mul = jax.lax.dynamic_update_index_in_dim(to_mul, rho, small_idx, 0)

            return [(q * m).astype(q.dtype) for q, m in zip(Q, to_mul)]

        key, subkey = jax.random.split(key)
        do_balances = jax.random.uniform(subkey, shape=(len(Qs),)) < 0.01
        Qs = [
            (
                jax.lax.cond(
                    jnp.logical_and(db, do_update),
                    vmap(balance_Q) if s else balance_Q,
                    lambda q: q,
                    Q,
                )
                if len(Qs) > 1
                else Q
            )
            for db, Q, s in zip(do_balances, Qs, flat_scanned_layers)
        ]

        # precondition gradients
        precond_gs = []
        for Q, expr, g, s in zip(Qs, expressions, updates, flat_scanned_layers):
            if s:
                precond_gs.append(
                    vmap(_precond_grad_kron_math, in_axes=(0, None, 0))(Q, expr, g)
                )
            else:
                precond_gs.append(_precond_grad_kron_math(Q, expr, g))

        # trust region
        # global clipping
        max_norm = jnp.sqrt(
            jnp.array(
                [p.size for p in jax.tree.leaves(precond_gs)], dtype=jnp.float32
            ).sum()
        )
        precond_gs = _global_clip(precond_gs, max_norm)
        # element-wise clipping
        precond_gs = jax.tree.map(lambda x: jnp.clip(x, -1.0, 1.0), precond_gs)

        # unflatten pytrees
        updates = grads_structure.unflatten(precond_gs)
        Qs = grads_structure.unflatten(Qs)

        # dtypes and new state
        mu = otu.tree_cast(mu, mu_dtype)
        Qs = otu.tree_cast(Qs, precond_dtype)
        state = dict(count=count_inc, key=key, mu=mu, Qs_preconditioners=Qs)

        return updates, state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def kron(
    learning_rate: Union[float, Callable[[int], float]] = 0.001,
    preconditioner_update_probability: Union[float, Callable[[int], float]] = 0.5,
    b1: float = 0.9,
    nesterov: bool = False,
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    max_size_triangular: int = 4096,
    max_skew_triangular: int = 10,
    precond_lr: Union[float, Callable[[int], float]] = 0.1,
    precond_init_scale: float = 0.1,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precision: str = "bfloat16",
    scanned_layers: Optional[base.Params] = None,
) -> base.GradientTransformationExtraArgs:
    """
    Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    Args:
        learning_rate: float or callable, learning rate.
        preconditioner_update_probability: float, probability of updating the
            preconditioner.
        b1: float, momentum parameter.
        nesterov: bool, whether to use Nesterov momentum.
        weight_decay: float, weight decay.
        mask: optional Any or callable, mask to apply to parameters.
        max_size_triangular: int, max size for preconditioner to be triangular.
        max_skew_triangular: int, max skew for preconditioner to be triangular.
        precond_lr: float or callable, learning rate for the preconditioner.
        precond_init_scale: float, initial scale for the preconditioner.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum accumulator.
            Defaults to the same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioner.
        precision: str, precision for matmul, 'bfloat16', 'tensorfloat32', 'float32'.
        scanned_layers: optional base.Params, tree of bool indicating scanned layers.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    opt = [
        scale_by_kron(
            preconditioner_update_probability=preconditioner_update_probability,
            b1=b1,
            nesterov=nesterov,
            max_size_triangular=max_size_triangular,
            max_skew_triangular=max_skew_triangular,
            precond_lr=precond_lr,
            precond_init_scale=precond_init_scale,
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
            precision=precision,
            scanned_layers=scanned_layers,
        )
    ]
    if weight_decay > 0:
        opt.append(transform.add_decayed_weights(weight_decay, mask=mask))
    opt.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*opt)


def _apply_momentum(
    updates: base.Updates, momentum: base.Updates, step, b1, nesterov
) -> Tuple[base.Updates, base.Updates]:
    # ema
    mu = otu.tree_update_moment(updates, momentum, b1, 1)
    if nesterov:
        # nesterov momentum for ema with bias correction
        # https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
        updates = jax.tree.map(
            lambda m, g: b1 * m + (1 - b1) * g,
            otu.tree_bias_correction(mu, b1, safe_int32_increment(step)),
            otu.tree_bias_correction(updates, b1, step),
        )
    else:
        # bias correction only
        updates = otu.tree_bias_correction(mu, b1, step)

    return updates, mu


def _add_eps(x):
    return jnp.clip(x, 1e-30, None)


def _global_clip(updates, max_norm):
    g_norm = global_norm(updates)
    g_norm = jnp.maximum(max_norm, g_norm)
    updates = jax.tree.map(
        lambda u: (u / g_norm.astype(u.dtype)) * max_norm.astype(u.dtype), updates
    )
    return updates


def _norm_lower_bound(A: jax.Array):
    """Returns a cheap lower bound for the spectral norm of A.

    Numerical results on random matrices with a wide range of distributions and
    sizes suggest, norm(A) <= sqrt(2) * norm_lower_bound(A). Looks to be a very
    tight lower bound.
    """
    max_abs = jnp.max(jnp.abs(A))

    def calc(A):
        A = A / max_abs

        aa = jnp.real(A * A.conj())

        aa_sum0 = jnp.sum(aa, axis=0)
        aa_sum1 = jnp.sum(aa, axis=1)
        i = jnp.argmax(aa_sum0, 0)
        j = jnp.argmax(aa_sum1, 0)
        value0 = jax.lax.dynamic_index_in_dim(aa_sum0, i, 0, keepdims=False)
        value1 = jax.lax.dynamic_index_in_dim(aa_sum1, j, 0, keepdims=False)

        def gt_branch():
            x = jax.lax.dynamic_index_in_dim(A, i, 1, keepdims=False)
            x = x.conj() @ A
            return max_abs * jnp.linalg.norm((x / jnp.linalg.norm(x)) @ A.conj().T)

        def le_branch():
            x = jax.lax.dynamic_index_in_dim(A, j, 0, keepdims=False)
            x = A @ x.conj()
            return max_abs * jnp.linalg.norm(A.conj().T @ (x / jnp.linalg.norm(x)))

        # these lax.cond fall back to lax.select inside vmap but we'll keep them anyway
        return jax.lax.cond(value0 > value1, gt_branch, le_branch)

    def no_calc(_):
        return max_abs

    return jax.lax.cond(max_abs > 0, calc, no_calc, A)


def _init_Q_exprs(t, scale, max_size, max_skew, dtype, existing_Q=None):
    """
    For a scalar or tensor `t`, we initialize its preconditioner `Q` and reusable
    contraction expressions for updating `Q` and preconditioning gradient.

    1, Preconditioner `Q` is initialized to
    `Q = scale * I = scale * kron(eye(t.shape[0]), eye(t.shape[1]), ...)`
    where the `eye(.)` may be replaced with `diag(ones(.))` if that dim is too large,
    determined by `max_size` and `max_skew`.

    2, A series of einsum contract expressions. The following subscript examples are for
    a 5th order tensor.
        2.1, `exprA` is the expression for calculating `A`, e.g.,
            `'aA,bB,cC,dD,eE,ABCDE->abcde'`
        2.2, `exprGs` is a list of expressions for calculating the gradients wrt `Q`
            on each dim, e.g., `'abCde,abγde->Cγ'` for the middle dim of a 5th order
            tensor `Q`.
        2.3, `exprP` is the expression for calculating the preconditioned gradient,
            e.g., `'aA,bB,cC,dD,eE,aα,bβ,cγ,dδ,eε,αβγδε->ABCDE'`
    """
    shape = t.shape
    if len(shape) == 0:  # scalar
        Q = (
            [scale * jnp.ones_like(t, dtype=dtype)]
            if existing_Q is None
            else existing_Q
        )
        exprA = opt_einsum.contract_expression(",->", Q[0].shape, t.shape)
        exprP = opt_einsum.contract_expression(",,->", Q[0].shape, Q[0].shape, t.shape)
        exprGs = [opt_einsum.contract_expression(",->", t.shape, t.shape)]
    else:  # tensor
        if len(shape) > 26:
            raise ValueError(
                f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters; "
                "Replace 26 with larger numbers!"
            )

        scale = scale ** (1 / len(shape))
        if len(shape) == 1:
            beta_size = 1  # 2nd largest size
        else:
            beta_size = sorted(list(shape))[-2]

        Q = [] if existing_Q is None else existing_Q
        exprGs = []
        # used for getting the subscripts for exprA
        piece1A, piece2A, piece3A = ([], "", "")
        # used for getting the subscripts for exprP
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
        for i, size in enumerate(shape):
            if size == 1 or size > max_size or size > max_skew * beta_size:
                # use diagonal matrix as preconditioner for this dim
                if existing_Q is None:
                    Q.append(scale * jnp.ones(size, dtype=dtype))

                piece1A.append(opt_einsum.get_symbol(i))
                piece2A = piece2A + opt_einsum.get_symbol(i)
                piece3A = piece3A + opt_einsum.get_symbol(i)

                piece1P.append(opt_einsum.get_symbol(i + 26))
                piece2P.append(opt_einsum.get_symbol(i + 26))
                piece3P = piece3P + opt_einsum.get_symbol(i + 26)
                piece4P = piece4P + opt_einsum.get_symbol(i + 26)

                piece1 = "".join(
                    [
                        (
                            opt_einsum.get_symbol(i + 26)
                            if j == i
                            else opt_einsum.get_symbol(j)
                        )
                        for j in range(len(shape))
                    ]
                )
                subscripts = (
                    piece1 + "," + piece1 + "->" + opt_einsum.get_symbol(i + 26)
                )
                exprGs.append(
                    opt_einsum.contract_expression(subscripts, t.shape, t.shape)
                )
            else:
                # use triangular matrix as preconditioner for this dim
                if existing_Q is None:
                    Q.append(scale * jnp.eye(size, dtype=dtype))

                piece1A.append(opt_einsum.get_symbol(i) + opt_einsum.get_symbol(i + 26))
                piece2A = piece2A + opt_einsum.get_symbol(i + 26)
                piece3A = piece3A + opt_einsum.get_symbol(i)

                a, b, c = (
                    opt_einsum.get_symbol(i),
                    opt_einsum.get_symbol(i + 26),
                    opt_einsum.get_symbol(i + 805),
                )
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

                piece1 = "".join(
                    [
                        (
                            opt_einsum.get_symbol(i + 26)
                            if j == i
                            else opt_einsum.get_symbol(j)
                        )
                        for j in range(len(shape))
                    ]
                )
                piece2 = "".join(
                    [
                        (
                            opt_einsum.get_symbol(i + 805)
                            if j == i
                            else opt_einsum.get_symbol(j)
                        )
                        for j in range(len(shape))
                    ]
                )
                subscripts = (
                    piece1
                    + ","
                    + piece2
                    + "->"
                    + opt_einsum.get_symbol(i + 26)
                    + opt_einsum.get_symbol(i + 805)
                )
                exprGs.append(
                    opt_einsum.contract_expression(subscripts, t.shape, t.shape)
                )

        subscripts = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprA = opt_einsum.contract_expression(
            subscripts, *[q.shape for q in Q], t.shape
        )

        subscripts = (
            ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
        )
        exprP = opt_einsum.contract_expression(
            subscripts, *[q.shape for q in Q], *[q.shape for q in Q], t.shape
        )

    exprGs = tuple(exprGs)

    if existing_Q is not None:
        return exprA, exprGs, exprP

    print(
        f"Q: {[q.shape for q in Q]}",
        f"exprA: {exprA}",
        f"exprGs: {exprGs}",
        f"exprP: {exprP}",
        sep="\n",
    )
    return [Q, (exprA, exprGs, exprP)]


def _update_precond_kron_math(Q, G, V, exprs, precond_lr, precision):
    """Update Kronecker product preconditioner Q with (vector, hess-vector-product).

    V is optional, and we can set it to None if it is integrated out (NOT recommended).
    """

    with jax.default_matmul_precision(precision):

        def solve_triangular(A, B, upper, left=True):
            leading_dims = 0
            if B.ndim > 2:
                leading_dims = B.ndim - 2
            solve_fn = partial(
                jax.lax.linalg.triangular_solve, left_side=left, lower=not upper
            )
            for _ in range(leading_dims):
                solve_fn = vmap(solve_fn, in_axes=(None, 0))
            return solve_fn(A, B)

        def triangular_inv(A):
            # return inv(A); used only when V is None, i.e., integrating out V
            I = jnp.eye(A.shape[0], dtype=A.dtype)
            return solve_triangular(A, I, upper=True)

        def solve_triangular_right(X, A):
            # return X @ inv(A)
            if X.ndim > 1:
                return solve_triangular(A, X, upper=True, left=False)
            else:
                return solve_triangular(A, X[None, :], upper=True, left=False)[0]

        order = G.ndim  # order of tensor

        exprA, exprGs, _ = exprs

        A = exprA(*Q, G, backend="jax")
        if V is not None:
            invQhinvQ, trace_invQhinvQ = None, None
            p = list(range(order))
            # permute dims like [0,1,2,3,4] -> [1,2,3,4,0]
            conjB = jnp.transpose(V.conj(), p[1:] + p[:1])
            for i, q in enumerate(Q):
                conjB = conjB / q if q.ndim < 2 else solve_triangular_right(conjB, q)
                if i < order - 1:
                    # transpose dims like
                    # [1,2,3,4,0]->[0,2,3,4,1]->[0,1,3,4,2]->[0,1,2,4,3]->[0,1,2,3,4]
                    conjB = jnp.swapaxes(conjB, i, order - 1)
        else:  # V is integrated out, and no need to form conjB
            conjB = None
            invQ = [1 / q if q.ndim < 2 else triangular_inv(q) for q in Q]
            invQhinvQ = [q.conj() * q if q.ndim < 2 else q.conj().T @ q for q in invQ]
            trace_invQhinvQ = [
                jnp.sum(q) if q.ndim < 2 else jnp.trace(q) for q in invQhinvQ
            ]

        def update_q(q, i):
            step_normalizer = "2nd"
            term1 = exprGs[i](A, A.conj())
            if conjB is not None:
                term2 = exprGs[i](conjB.conj(), conjB)
            else:  # V is integrated out
                term2 = 1.0
                for j, trace in enumerate(trace_invQhinvQ):
                    term2 = term2 * (trace if i != j else invQhinvQ[i])

            if step_normalizer == "2nd":
                if q.ndim < 2:  # q is a diagonal matrix or scalar
                    q -= (
                        precond_lr
                        / _add_eps(jnp.max(jnp.abs(term1 + term2)))
                        * (term1 - term2)
                        * q
                    )
                else:
                    q -= (
                        precond_lr
                        / _add_eps(_norm_lower_bound(term1 + term2))
                        * jnp.triu(term1 - term2)
                        @ q
                    )
            else:  # only use gradient for step size normalization
                if q.ndim < 2:  # q is a diagonal matrix or scalar
                    q -= (
                        precond_lr
                        / _add_eps(jnp.max(jnp.abs(term1 - term2)))
                        * (term1 - term2)
                        * q
                    )
                else:
                    q -= (
                        precond_lr
                        / _add_eps(_norm_lower_bound(term1 - term2))
                        * jnp.triu(term1 - term2)
                        @ q
                    )
            return q

        Q = [update_q(q, i) for i, q in enumerate(Q)]

        return Q


def _precond_grad_kron_math(Q, exprs, G):
    """Precondition gradient G with preconditioner Q."""
    # the last expr is exprP
    return exprs[-1](*[q.conj() for q in Q], *Q, G, backend="jax")


if __name__ == "__main__":
    from pprint import pprint

    params = {
        "a": jnp.ones(()),
        "b": jnp.ones((2,)),
        "c": jnp.ones((2, 100)),
        "d": jnp.ones((3, 7, 4)),
        "e": jnp.ones((2, 3, 4)),  # scan
        "f": jnp.ones((2, 4, 5, 2)),  # scan
        "g": jnp.ones((2, 5)),  # scan
        "h": jnp.ones((1, 2, 3, 4, 4, 3)),
    }
    scanned = {
        "a": False,
        "b": False,
        "c": False,
        "d": False,
        "e": True,
        "f": True,
        "g": True,
        "h": False,
    }
    print("Params:")
    pprint(jax.tree.map(lambda p: p.shape, params), width=150)
    pprint(jnp.array([x.mean() for x in jax.tree.leaves(params)]).mean())

    opt = kron(learning_rate=0.1, scanned_layers=scanned)
    opt_state = jax.jit(opt.init)(params)
    print("Opt State:")
    pprint(jax.tree.map(lambda p: p.shape, opt_state), width=150)

    for _ in range(10):
        grads = jax.tree.map(lambda p: p * 2, params)
        updates, opt_state = jax.jit(opt.update)(grads, opt_state)
        params = jax.tree.map(lambda p, u: p + u, params, updates)

    print("Params:")
    pprint(jax.tree.map(lambda p: p.shape, params), width=150)
    pprint(jnp.array([x.mean() for x in jax.tree.leaves(params)]).mean())
