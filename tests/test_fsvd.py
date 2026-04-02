"""Tests for PtyLabX.utils.fsvd — Randomized SVD."""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from PtyLabX.utils.fsvd import ortho_basis, rsvd, subspace_iter


def test_ortho_basis_orthogonality():
    """ortho_basis should produce orthonormal columns."""
    rng = np.random.default_rng(0)
    M = jnp.array(rng.standard_normal((50, 10)), dtype=jnp.float32)
    Q = ortho_basis(M)
    # Q^T Q should be identity
    QtQ = np.asarray(Q.T @ Q)
    assert_allclose(QtQ, np.eye(10), atol=2e-4)


def test_rsvd_approx_matches_full_svd():
    """Rank-k rsvd approximation should be close to full SVD for a low-rank matrix."""
    rng = np.random.default_rng(1)
    # Create a rank-3 matrix
    A_np = rng.standard_normal((50, 5)) @ rng.standard_normal((5, 40))
    A = jnp.array(A_np, dtype=jnp.float32)

    U, S, Vt = rsvd(A, rank=5, rng_key=jax.random.PRNGKey(42))
    A_approx = U @ jnp.diag(S) @ Vt
    # Reconstruction error should be small
    error = float(jnp.linalg.norm(A - A_approx) / jnp.linalg.norm(A))
    assert error < 0.01


def test_subspace_iter_improves_approximation():
    """Subspace iteration should improve the range approximation."""
    rng = np.random.default_rng(2)
    A = jnp.array(rng.standard_normal((40, 30)) + 1j * rng.standard_normal((40, 30)), dtype=jnp.complex64)

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    Y0 = jax.random.normal(k1, shape=(40, 10)) + 1j * jax.random.normal(k2, shape=(40, 10))
    Y0 = Y0.astype(jnp.complex64)

    Q0 = ortho_basis(Y0)
    Q_iter = subspace_iter(A, Y0, 3)

    # Projection error should decrease after iteration
    err0 = float(jnp.linalg.norm(A - Q0 @ (Q0.T.conj() @ A)))
    err_iter = float(jnp.linalg.norm(A - Q_iter @ (Q_iter.T.conj() @ A)))
    assert err_iter <= err0
