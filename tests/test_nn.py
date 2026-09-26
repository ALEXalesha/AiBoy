import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nn import io, layers, losses, rnn
from nn.optim import Adam, clip_grads


def numeric_grad(f, x, eps=1e-5):
    """Центральная разность по каждому элементу x (x меняется на месте и возвращается)."""
    g = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        up = f()
        x[i] = old - eps
        down = f()
        x[i] = old
        g[i] = (up - down) / (2 * eps)
    return g


def check_layer(layer, x, rng):
    """backward слоя против численного градиента по входу и по всем весам. Потеря - сумма
    выхода со случайными коэффициентами, чтобы градиенты были разными."""
    y = layer.forward(x)
    coef = rng.normal(size=y.shape)

    def loss():
        return float((layer.forward(x) * coef).sum())

    layer.forward(x)
    dx = layer.backward(coef)
    assert dx.shape == x.shape
    np.testing.assert_allclose(dx, numeric_grad(loss, x), rtol=1e-5, atol=1e-7)
    layer.forward(x)
    layer.backward(coef)
    for name, p, grad in layer.params():
        np.testing.assert_allclose(grad, numeric_grad(loss, p), rtol=1e-5, atol=1e-7, err_msg=name)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_dense_gradients(rng):
    check_layer(layers.Dense(5, 3, rng, dtype=np.float64), rng.normal(size=(4, 5)), rng)


def test_dense_works_on_sequences(rng):
    check_layer(layers.Dense(4, 3, rng, dtype=np.float64), rng.normal(size=(2, 5, 4)), rng)


def test_relu_gradients(rng):
    x = rng.normal(size=(3, 4))
    x[np.abs(x) < 0.05] = 0.5  # подальше от излома
    check_layer(layers.ReLU(), x, rng)


def test_tanh_and_sigmoid_gradients(rng):
    check_layer(layers.Tanh(), rng.normal(size=(3, 4)), rng)
    check_layer(layers.Sigmoid(), rng.normal(size=(3, 4)) * 3, rng)


def test_sequential_gradients(rng):
    net = layers.Sequential([layers.Dense(4, 6, rng, dtype=np.float64), layers.Tanh(),
                             layers.Dense(6, 2, rng, dtype=np.float64), layers.Sigmoid()])
    check_layer(net, rng.normal(size=(3, 4)), rng)
    assert [n for n, _, _ in net.params()] == ["0.W", "0.b", "2.W", "2.b"]


def test_rnn_cell_gradients_by_input_state_and_weights(rng):
    cell = rnn.RNNCell(3, 4, rng, dtype=np.float64)
    x = rng.normal(size=(2, 3))
    h = rng.normal(size=(2, 4)) * 0.5
    coef = rng.normal(size=(2, 4))

    def loss():
        return float((cell.forward(x, h) * coef).sum())

    cell.forward(x, h)
    dx, dh = cell.backward(coef)
    np.testing.assert_allclose(dx, numeric_grad(loss, x), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(dh, numeric_grad(loss, h), rtol=1e-5, atol=1e-7)
    cell.forward(x, h)
    cell.backward(coef)
    for name, p, grad in cell.params():
        np.testing.assert_allclose(grad, numeric_grad(loss, p), rtol=1e-5, atol=1e-7, err_msg=name)


def test_gru_gradients_through_time(rng):
    """Обратный проход во времени: по входам всех шагов, по начальному состоянию и весам."""
    gru = rnn.GRU(3, 4, rng, dtype=np.float64)
    x = rng.normal(size=(2, 5, 3))
    h0 = rng.normal(size=(2, 4)) * 0.5
    coef = rng.normal(size=(2, 5, 4))

    def loss():
        return float((gru.forward(x, h0) * coef).sum())

    gru.forward(x, h0)
    dx, dh0 = gru.backward(coef)
    assert dx.shape == x.shape and dh0.shape == h0.shape
    np.testing.assert_allclose(dx, numeric_grad(loss, x), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(dh0, numeric_grad(loss, h0), rtol=1e-5, atol=1e-7)
    gru.forward(x, h0)
    gru.backward(coef)
    for name, p, grad in gru.params():
        np.testing.assert_allclose(grad, numeric_grad(loss, p), rtol=1e-5, atol=1e-7, err_msg=name)


def test_gru_step_matches_the_sequence(rng):
    gru = rnn.GRU(3, 4, rng, dtype=np.float64)
    x = rng.normal(size=(2, 6, 3))
    h = rng.normal(size=(2, 4))
    seq = gru.forward(x, h)
    for t in range(6):
        h = gru.step(x[:, t], h)
        np.testing.assert_allclose(h, seq[:, t], rtol=1e-12)


def test_softmax_xent_gradient_with_weights(rng):
    logits = rng.normal(size=(3, 4, 5))
    target = rng.integers(0, 5, size=(3, 4))
    weight = (rng.random((3, 4)) > 0.3).astype(float)
    _, grad = losses.softmax_xent(logits, target, weight)
    num = numeric_grad(lambda: losses.softmax_xent(logits, target, weight)[0], logits)
    np.testing.assert_allclose(grad, num, rtol=1e-5, atol=1e-8)


def test_mse_gradient(rng):
    pred = rng.normal(size=(4, 3))
    target = rng.normal(size=(4, 3))
    _, grad = losses.mse(pred, target)
    np.testing.assert_allclose(grad, numeric_grad(lambda: losses.mse(pred, target)[0], pred), rtol=1e-6)


def test_adam_fits_a_line(rng):
    net = layers.Dense(1, 1, rng, dtype=np.float64)
    opt = Adam(net.params(), lr=0.05)
    x = np.linspace(-1, 1, 32)[:, None]
    for _ in range(600):
        _, g = losses.mse(net.forward(x), 3 * x - 1)
        net.backward(g)
        opt.step()
    assert abs(net.W[0, 0] - 3) < 0.02 and abs(net.b[0] + 1) < 0.02


def test_clip_grads_limits_the_total_norm(rng):
    net = layers.Dense(3, 3, rng, dtype=np.float64)
    net.dW[...] = 100.0
    net.db[...] = 100.0
    norm = clip_grads(net.params(), 1.0)
    assert norm > 1.0
    total = np.sqrt(sum(float((g * g).sum()) for _, _, g in net.params()))
    assert total == pytest.approx(1.0)


@given(st.floats(min_value=-50, max_value=50), st.floats(min_value=-50, max_value=50))
def test_losses_stay_finite_on_big_logits(a, b):
    logits = np.array([[a, b, 0.0]])
    loss, grad = losses.softmax_xent(logits, np.array([0]))
    assert np.isfinite(loss) and np.isfinite(grad).all()


def test_save_and_load_roundtrip(tmp_path, rng):
    net = {"a": layers.Dense(3, 2, rng), "g": rnn.GRU(2, 3, rng)}
    path = tmp_path / "w.npz"
    io.save(path, io.collect(net))
    other = {"a": layers.Dense(3, 2, np.random.default_rng(5)), "g": rnn.GRU(2, 3, np.random.default_rng(5))}
    io.restore(other, io.load(path))
    for name in net:
        for (_, p, _), (_, q, _) in zip(net[name].params(), other[name].params()):
            np.testing.assert_array_equal(p, q)


def test_restore_refuses_wrong_shapes(tmp_path, rng):
    io.save(tmp_path / "w.npz", io.collect({"a": layers.Dense(3, 2, rng)}))
    with pytest.raises(ValueError):
        io.restore({"a": layers.Dense(4, 2, rng)}, io.load(tmp_path / "w.npz"))
