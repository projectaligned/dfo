import itertools
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

# The real figure shows the input and output of the function we are trying to approximate
real_fig = plt.figure()
real_ax = real_fig.add_subplot(111, projection='3d')
real_ax.set_xlim(0, 1)
real_ax.set_ylim(0, 1)
real_ax.set_zlim(0, 1)


# The weight figure shows the weights and the losses for the approximating function
weights_fig = plt.figure()
weights_ax = weights_fig.add_subplot(111, projection='3d')
weights_ax.set_xlim(-20, 20)
weights_ax.set_xlabel('xweights')
weights_ax.set_ylim(-20, 20)
weights_ax.set_ylabel('yweights')
weights_ax.set_zlim(-20, 20)
weights_ax.set_zlabel('biases')

sample_fig = plt.figure()
sample_ax = sample_fig.add_subplot(111)
sample_ax.set_xlabel('num samples')
sample_ax.set_ylabel('loss')
sample_ax.set_yscale('log')

Func = Callable[[np.ndarray], float]


def xor(point: np.ndarray) -> float:
    assert point.shape == (2,)
    return float(point[0] ^ point[1])


def ior(point: np.ndarray) -> float:
    assert point.shape == (2,)
    return float(point[0] | point[1])


def b_and(point: np.ndarray) -> float:
    assert point.shape == (2,)
    return float(point[0] & point[1])


def plot_separator(weights: np.ndarray) -> None:
    assert len(weights.shape) == 1
    xx, yy = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    point = np.array([xx, yy, 1])
    # divide by a small value in order to plot the hyperplane.
    z = (-np.dot(weights, point)) / 0.01
    real_ax.plot_surface(xx, yy, z)


def boolean_points(dim: int = 2) -> np.ndarray:
    return np.array(list(itertools.product(*(dim*[[0, 1]]))))


def plot_func(func: Func) -> None:
    xys = boolean_points(dim=2)
    zs = [func(xy) for xy in xys]
    xs, ys = xys.T
    real_ax.scatter(xs, ys, zs)


def build_linear_model(weights: np.ndarray) -> Func:
    assert len(weights.shape) == 1

    def linear_model(point: np.ndarray) -> float:
        point = np.append(point, 1)
        assert point.shape == weights.shape
        return logistic(-np.dot(weights, point))
    return linear_model


def logistic(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def compute_loss(points: np.ndarray, func: Func, predictor: Func) -> (float, float):
    assert points.shape[1] == 2
    evals = [(func(point), predictor(point)) for point in points]
    l1 = sum([abs(actual - predicted) for (actual, predicted) in evals])
    l2 = np.sqrt(sum((actual - predicted)**2 for (actual, predicted) in evals))
    return l1, l2


def test_linear(weights: np.ndarray) -> (float, float):
    linear_model = build_linear_model(weights)
    points = boolean_points(dim=2)
    l1, l2 = compute_loss(points, ior, linear_model)
    return l1, l2


def random_search(bool_func: Func, trials: int = 1000):
    points = boolean_points(dim=2)
    evals = []
    for trial in range(trials):
        weights = np.random.uniform(-20, 20, 3)
        linear_model = build_linear_model(weights)
        l1, l2 = compute_loss(points, bool_func, linear_model)
        evals.append([l1, l2, weights])

    best_l1_eval = min(evals, key=lambda x: x[0])
    best_l2_eval = min(evals, key=lambda x: x[1])
    print(best_l1_eval)
    print(best_l2_eval)
    l1s, l2s, weights = zip(*evals)
    return np.array(l1s), np.array(l2s), np.array(weights)


def cross_entropy_optimization(bool_func: Func, max_trials: int = 100, num_evaluate: int = 100, num_update: int = 10):
    """
    using the cross-entropy-method https://en.wikipedia.org/wiki/Cross-entropy_method
    to minimize the loss.
    """
    points = boolean_points(dim=2)

    weights_x_mean = 0
    weights_x_std = 10
    weights_y_mean = 0
    weights_y_std = 10
    biases_mean = 0
    biases_std = 10

    all_results = []
    n_trial = 0
    while n_trial < max_trials:
        if weights_x_std < 0.01 and weights_y_std < 0.01 and biases_std < 0.01:
            break
        weights_x_samples = np.random.normal(weights_x_mean, weights_x_std, num_evaluate)
        weights_y_samples = np.random.normal(weights_y_mean, weights_y_std, num_evaluate)
        biases_samples = np.random.normal(biases_mean, biases_std, num_evaluate)

        evaluation_results = []
        for eval_idx in range(num_evaluate):
            weights = np.array([weights_x_samples[eval_idx], weights_y_samples[eval_idx], biases_samples[eval_idx]])
            linear_model = build_linear_model(weights)
            l1, l2 = compute_loss(points, bool_func, linear_model)
            evaluation_results.append([l1, l2, weights])
            all_results.append([l1, l2, weights])

        # optimize using l1
        best_results = sorted(evaluation_results, key=lambda result: result[1])
        top_results = best_results[:num_update]

        weights = np.array([result[2] for result in top_results])
        weights_x, weights_y, biases = weights.T
        weights_x_mean = np.mean(weights_x)
        weights_x_std = np.std(weights_x)
        weights_y_mean = np.mean(weights_y)
        weights_y_std = np.std(weights_y)
        biases_mean = np.mean(biases)
        biases_std = np.std(biases)

        n_trial += 1

    print(weights_x_std)
    print(weights_y_std)
    print(biases_std)
    print(n_trial)
    best_l1_eval = min(all_results, key=lambda x: x[0])
    best_l2_eval = min(all_results, key=lambda x: x[1])
    print(best_l1_eval)
    print(best_l2_eval)
    l1s, l2s, weights = zip(*all_results)
    return np.array(l1s), np.array(l2s), np.array(weights), best_l1_eval, best_l2_eval


def cma_es(bool_func: Func, max_trials: int = 100, samples_per_iteration: int = 30, num_update: int = 10):
    """
    This function performs the CMA-ES https://en.wikipedia.org/wiki/CMA-ES optimization algorithm
    """

    points = boolean_points(dim=2)

    step_size = 0.1
    weights_mean = np.zeros((3,))
    covariance_matrix = np.identity(3)
    iso_path = np.zeroes((3,))
    aniso_path = np.zeroes((3,))

    all_results = []
    n_trials = 0
    while n_trials < max_trials:
        weights_samples = np.random.multivariate_normal(mean=weights_mean, cov=covariance_matrix,
                                                        size=samples_per_iteration)

        evaluation_results = []
        for weights_sample in weights_samples:
            linear_model = build_linear_model(weights_sample)
            l1, l2 = compute_loss(points, bool_func, linear_model)
            evaluation_results.append([l1, l2, weights])
            all_results.append([l1, l2, weights])

        # optimize using l1
        best_results = sorted(evaluation_results, key=lambda result: result[0])
        selected_results = best_results[:num_update]
        selected_l1s, selected_l2s, selected_weights = zip(*selected_results)
        selected_weights_array = np.array(selected_weights)
        new_mean = np.mean(np.array(selected_weights), axis=0)


def visualize_loss(losses: np.ndarray, weights: np.ndarray) -> None:
    xs, ys, zs = weights.T
    weights_ax.scatter(xs, ys, zs, c=losses, cmap='viridis')


def visualize_sample_efficiency(losses: np.ndarray) -> None:
    sample_ax.plot(losses)


#l1s, l2s, weights = random_search(ior)
l1s, l2s, weights, best_l1_eval, best_l2_eval = cross_entropy_optimization(ior)
visualize_loss(l1s, weights)
visualize_sample_efficiency(l1s)
plot_func(ior)
_, _, best_l1_weights = best_l1_eval
plot_separator(best_l1_weights)
plt.show()

