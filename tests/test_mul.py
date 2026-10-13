"""
Sanity tests for the hand-written NumPy library.

    python -m pytest tests -q        (or simply: python tests/test_mul.py)

1. Gradient checks: backprop vs finite differences, for every loss used by an unlearning method.
2. SISA exactness: unlearning == training from scratch without the point, bit for bit.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mul import methods, train  # noqa: E402
from mul.nn import MLP, cross_entropy, distill_kl  # noqa: E402


def _numeric_grad(f, P, eps=1e-3, n_checks=25, rng=None):
    """Central finite differences on a few random coordinates of parameter array P."""
    rng = rng or np.random.default_rng(0)
    out = []
    for _ in range(n_checks):
        i = tuple(rng.integers(0, s) for s in P.shape)
        old = P[i]
        P[i] = old + eps
        up = f()
        P[i] = old - eps
        down = f()
        P[i] = old
        out.append((i, (up - down) / (2 * eps)))
    return out


def _check(loss_and_grad, model, X):
    model.params = [p.astype(np.float64) for p in model.params]
    _, g = loss_and_grad(model.forward(X, keep_cache=True))
    grads = model.backward(g)
    grads = [gr.astype(np.float64) for gr in grads]
    for P, G in zip(model.params, grads):
        for idx, num in _numeric_grad(lambda: loss_and_grad(model.forward(X))[0], P):
            assert abs(num - G[idx]) < 1e-3 + 1e-2 * abs(num), (num, G[idx])


def test_cross_entropy_gradient():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(16, 5)), rng.integers(0, 3, 16)
    _check(lambda z: cross_entropy(z, y), MLP(5, 3, (8, 8), seed=1), X)


def test_distillation_gradient():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(16, 5))
    teacher = rng.normal(size=(16, 3))
    for T in (1.0, 4.0):
        _check(lambda z: distill_kl(z, teacher, T), MLP(5, 3, (8,), seed=2), X)


def test_sisa_is_exact():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(120, 6)).astype(np.float32), rng.integers(0, 3, 120)
    fn = lambda: MLP(6, 3, (16,))
    s = methods.SISA(fn, n_shards=3, n_slices=3, epochs_per_slice=3, seed=7)
    s.fit(X, y)
    forget = [5, 17, 60]
    s.unlearn(forget)
    scratch = methods.SISA(fn, n_shards=3, n_slices=3, epochs_per_slice=3, seed=7)
    scratch.fit(X, y, partition=s.slices)            # same layout, the 3 points never present
    for a, b in zip(s.shard_models, scratch.shard_models):
        assert train.weights_equal(a, b)




# ---------------------------------------------------------------------------------------------
# Classical models: unlearning must equal refitting from scratch without the rows
# ---------------------------------------------------------------------------------------------
from mul import classical  # noqa: E402


def _toy(n=300, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y


def test_gaussian_nb_exact():
    X, y = _toy()
    f = np.arange(0, 300, 7); r = np.setdiff1d(np.arange(300), f)
    a = classical.GaussianNB().fit(X, y).unlearn(X[f], y[f])
    b = classical.GaussianNB().fit(X[r], y[r])
    assert np.allclose(a.predict_log_proba(X), b.predict_log_proba(X), atol=1e-8)


def test_multinomial_nb_exact():
    rng = np.random.default_rng(1)
    X, y = rng.poisson(1.0, size=(200, 30)), rng.integers(0, 2, 200)
    f = np.arange(0, 200, 9); r = np.setdiff1d(np.arange(200), f)
    a = classical.MultinomialNB().fit(X, y).unlearn(X[f], y[f])
    b = classical.MultinomialNB().fit(X[r], y[r])
    assert np.allclose(a.predict_log_proba(X), b.predict_log_proba(X), atol=1e-10)


def test_ridge_exact():
    X, y = _toy()
    f = np.arange(0, 300, 11); r = np.setdiff1d(np.arange(300), f)
    a = classical.RidgeUnlearner(1.0).fit(X, y).unlearn(X[f], y[f])
    b = classical.RidgeUnlearner(1.0).fit(X[r], y[r])
    assert np.allclose(a.w, b.w, atol=1e-8)
    Y = np.eye(2)[y]                                   # multi-output version
    a = classical.RidgeUnlearner(1.0).fit(X, Y).unlearn(X[f], Y[f])
    b = classical.RidgeUnlearner(1.0).fit(X[r], Y[r])
    assert np.allclose(a.w, b.w, atol=1e-8)


def test_knn_exact():
    X, y = _toy()
    f = np.arange(0, 300, 5); r = np.setdiff1d(np.arange(300), f)
    a = classical.KNN(5).fit(X, y).unlearn(f)
    b = classical.KNN(5).fit(X[r], y[r])
    assert (a.predict(X) == b.predict(X)).all()


def test_forest_exact():
    X, y = _toy()
    f = [3, 50, 120]
    a = classical.BaggedForest(n_trees=20, seed=3).fit(X, y)
    a.unlearn(f)
    b = classical.BaggedForest(n_trees=20, seed=3).fit(X, y, bags=[bg.copy() for bg in a.bags])
    assert np.allclose(a.predict_proba(X), b.predict_proba(X))


def test_logistic_newton_close():
    X, y = _toy()
    f = np.arange(0, 300, 10); r = np.setdiff1d(np.arange(300), f)
    full = classical.LogisticNewton(1.0).fit(X, y)
    approx = full.unlearn_newton(X[r], y[r], X[f], y[f])
    exact = classical.LogisticNewton(1.0).fit(X[r], y[r])
    # approximate: the residual is second-order, so allow 15% of the distance the deletion moved w
    assert np.linalg.norm(approx.w - exact.w) < 0.15 * np.linalg.norm(full.w - exact.w) + 1e-6


def test_lora_gradient_and_isolation():
    """LoRA: backward matches finite differences, a fresh adapter changes nothing, the base stays frozen."""
    from mul.nn import LoRA
    rng = np.random.default_rng(0)
    base = MLP(6, 3, (5,), seed=0)
    base_before = [p.copy() for p in base.params]
    X = rng.normal(size=(8, 6))
    y = rng.integers(0, 3, 8)
    lora = LoRA(base, rank=2, seed=1)
    assert np.allclose(lora(X), base(X))                              # B = 0 -> identical to the base
    for p in lora.params[1::2]:
        p += rng.normal(scale=0.3, size=p.shape).astype(np.float32)
    _check(lambda logits: cross_entropy(logits, y), lora, X)
    train.fit(lora, X, y, epochs=3, lr=1e-2, batch_size=4)
    assert all(np.array_equal(a, b) for a, b in zip(base.params, base_before))   # base untouched
    assert np.allclose(lora.merged()(X), lora(X), atol=1e-5)          # merging is the same function


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok ", name)
    print("all checks passed")

