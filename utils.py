import collections
import itertools
from multiprocessing.pool import ThreadPool
from typing import Mapping
import dataclasses
import numpy as np

import jax
import flax


def _traverse_with_names(tree, with_inner_nodes=False):
    """Traverses nested dicts/dataclasses and emits (leaf_name, leaf_val)."""
    if dataclasses.is_dataclass(tree):
        tree = flax.serialization.to_state_dict(tree)
    # Don't output the non-leaf nodes. If the optimizer doesn't have a state
    # the tree leaves can be Nones which was interpreted as a leaf by this
    # function but not by the other functions (like jax.tree.map).
    if tree is None:
        return
    elif isinstance(tree, Mapping):
        keys = sorted(tree.keys())
        for key in keys:
            for path, v in _traverse_with_names(tree[key], with_inner_nodes):
                yield (key + "/" + path).rstrip("/"), v
        if with_inner_nodes:
            yield "", tree
    elif isinstance(tree, (list, tuple)):
        for idx in range(len(tree)):
            for path, v in _traverse_with_names(tree[idx], with_inner_nodes):
                yield (str(idx) + "/" + path).rstrip("/"), v
        if with_inner_nodes:
            yield "", tree
    else:
        yield "", tree


def tree_flatten_with_names(tree):
    """Populates tree_flatten with leaf names.

    This function populates output of tree_flatten with leaf names, using a
    custom traversal that produces names is provided. The custom traversal does
    NOT have to traverse tree in the same order as jax, as we take care of
    automatically aligning jax' and custom traversals.

    Args:
      tree: python tree.

    Returns:
      A list of values with names: [(name, value), ...]
    """
    vals, tree_def = jax.tree.flatten(tree)

    # "Fake" token tree that is use to track jax internal tree traversal and
    # adjust our custom tree traversal to be compatible with it.
    tokens = range(len(vals))
    token_tree = tree_def.unflatten(tokens)
    val_names, perm = zip(*_traverse_with_names(token_tree))
    inv_perm = np.argsort(perm)

    # Custom traverasal should visit the same number of leaves.
    assert len(val_names) == len(vals)

    return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def


def make_fsarray_from_local_slice(local_slice, global_devices):
    """Create a fully-sharded global device array from local host arrays.

    Args:
      local_slice: Something convertible to a numpy array (eg also TF tensors)
        that is this host's slice of the global array.
      global_devices: The list of global devices. Needed for consistent ordering.

    Returns:
      The global on-device array which consists of all local slices stacked
      together in the order consistent with the devices.
    """
    mesh = jax.sharding.Mesh(global_devices, ("devices",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("devices"))
    local_ds = mesh.local_devices

    x = np.asarray(memoryview(local_slice))  # No-copy: http://(internal link)
    xs = jax.device_put(np.split(x, len(local_ds), axis=0), local_ds)

    global_shape = (x.shape[0] * jax.process_count(), *x.shape[1:])
    return jax.make_array_from_single_device_arrays(global_shape, sharding, xs)


def threadstart_iterator(it):
    """Starts an iterator right away in a background thread."""
    # We already want to "start" the iterator in order to start the underlying
    # dataset prefetch mechanisms, so here we get the first element. But we don't
    # want to lose it from training, so we yield that one afterwards.
    # (internal link)
    pool = ThreadPool(processes=1)
    first_ex_promise = pool.apply_async(lambda: next(it))

    yield first_ex_promise.get()
    yield from it


def prefetch_iterator(it, n):
    """Runs iterator `it` ahead for `n` steps. Adapted from flax."""
    if not n:
        yield from it
        return
    queue = collections.deque()

    def enqueue(n_steps):  # Enqueues *up to* `n` elements from the iterator.
        for data in itertools.islice(it, n_steps):
            # Prefetching will parallelize any processing that happens in a different
            # thread (like `jax.device_put()`), but it will be of no use for
            # processing that happens in the same thread.
            queue.append(data)

    enqueue(n)  # Fill up the buffer.
    while queue:
        yield queue.popleft()
        enqueue(1)


def tree_broadcast(prefix, target):
    """Broadcasts a prefix tree to a full tree.

    Input-output examples:
    1. prefix: {"x": 10, "y": 20}
       target: {"x": {"a": 1, "b": 2}, "y": 3}

       Result: {"x": {"a": 10, "b": 10}, "y": 20}

    2. prefix: 100
       target: {"x": {"a": 1, "b": 2}, "y": 3}

       Result: {"x": {"a": 100, "b": 100}, "y": 100}

    3. prefix: {"x": 10}
       target: {"x": {"a": 1, "b": 2}, "y": 3}

       Result: ValueError

    Args:
      prefix: prefix pytree.
      target: boradcast target for a prefix tree.

    Returns:
      prefix tree broadcasted to a target tree.
    """

    def _broadcast(leaf, subtree):
        return jax.tree.map(lambda _: leaf, subtree)

    return jax.tree.map(_broadcast, prefix, target)


def reshard(tree, shardings):
    """Take an arbitrarily* sharded pytree and shard it according to `shardings`.

    This is a no-op for tree elements which are already sharded as requested.

    *Arrays that are fully addressable (for example, CPU arrays) are assumed to be
    identical (i.e. replicated) across hosts.

    *It does not work if an element of `tree` is not fully-addressable, unless its
    sharding is already consistent with the target sharding.
    If this is needed, please ping lbeyer@ or akolesnikov@.

    Args:
      tree: a pytree of arrays.
      shardings: a (prefix) pytree of jax array shardings.
    Returns:
      A pytree of global jax arrays that follows provided shardings.
    """

    def _make_global_arr(x, shard, shape):
        # Avoid unnecessary copies and transfers:
        if hasattr(x, "sharding") and x.sharding.is_equivalent_to(
            shard, len(shape)
        ):  # pylint: disable=line-too-long
            return x
        if not getattr(x, "is_fully_addressable", True):
            raise RuntimeError(
                "Trying to reshard a non-fully-addressable array. "
                "Please see the doc-comment for detailed explanation."
            )
        x = jax.device_get(x)  # Might be on local devices.
        xs = [
            jax.device_put(x[s], device=d)
            for d, s in shard.addressable_devices_indices_map(shape).items()
        ]
        return jax.make_array_from_single_device_arrays(shape, shard, xs)

    shapes = jax.tree.map(np.shape, tree)
    shardings = tree_broadcast(shardings, tree)
    return jax.tree.map(_make_global_arr, tree, shardings, shapes)


def write_note(note: str):
    if jax.process_index() == 0:
        print(note)
