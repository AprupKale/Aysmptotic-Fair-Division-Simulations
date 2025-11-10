from __future__ import annotations
from dataclasses import dataclass
from typing import List, Callable, Optional
import numpy as np

# New: SciPy bits
from scipy.stats import truncnorm, johnsonsb
from scipy.special import expit, logit

# RNG wrapper -------------------------------------------------

@dataclass
class RNG:
    seed: Optional[int] = None
    def __post_init__(self):
        self._g = np.random.default_rng(self.seed)
    def uniform(self, a=0.0, b=1.0, size=None):
        return self._g.uniform(a, b, size)

# Base class --------------------------------------------------

class ItemDist:
    """Base class for an item's distribution D_j over [0,1]."""
    def sample_for_all_agents(self, n: int, rng: RNG) -> np.ndarray:
        raise NotImplementedError

class UniformItem(ItemDist):
    def __init__(self, a: float = 0.0, b: float = 1.0):
        assert 0.0 <= a < b <= 1.0
        self.a, self.b = a, b
    def sample_for_all_agents(self, n: int, rng: RNG) -> np.ndarray:
        return rng.uniform(self.a, self.b, size=n)

class BetaItem(ItemDist):
    """u(i,j) ~ Beta(alpha, beta) in [0,1]."""
    def __init__(self, alpha: float = 2.0, beta: float = 5.0):
        assert alpha > 0 and beta > 0
        self.alpha, self.beta = alpha, beta
    def sample_for_all_agents(self, n: int, rng: RNG) -> np.ndarray:
        # simpler & faster: use NumPy's beta
        return rng._g.beta(self.alpha, self.beta, size=n)

class TruncatedNormalItem(ItemDist):
    """
    Proper renormalized truncated Normal N(mu, sigma) on [0,1].
    Uses SciPy's truncnorm.rvs with random_state hooked to our RNG.
    """
    def __init__(self, mu: float = 0.5, sigma: float = 0.1):
        assert 0.0 <= mu <= 1.0 and sigma > 0
        self.mu, self.sigma = mu, sigma
    def sample_for_all_agents(self, n: int, rng: RNG) -> np.ndarray:
        a = (0.0 - self.mu) / self.sigma
        b = (1.0 - self.mu) / self.sigma
        return truncnorm.rvs(a, b, loc=self.mu, scale=self.sigma, size=n, random_state=rng._g)

class CallableItem(ItemDist):
    """User-provided sampler: fn(n, rng) -> shape (n,) in [0,1]."""
    def __init__(self, fn: Callable[[int, RNG], np.ndarray]):
        self.fn = fn
    def sample_for_all_agents(self, n: int, rng: RNG) -> np.ndarray:
        v = self.fn(n, rng)
        v = np.asarray(v, dtype=float)
        assert v.shape == (n,)
        return np.clip(v, 0.0, 1.0)

class KumaraswamyItem(ItemDist):
    """Kumaraswamy(a,b): CDF F(x) = 1 - (1 - x^a)^b on [0,1]."""
    def __init__(self, a: float = 2.0, b: float = 2.0):
        assert a > 0 and b > 0
        self.a, self.b = a, b
    def sample_for_all_agents(self, n: int, rng: RNG) -> np.ndarray:
        u = rng.uniform(0.0, 1.0, size=n)
        return (1.0 - (1.0 - u) ** (1.0 / self.b)) ** (1.0 / self.a)

class DeterministicItem(ItemDist):
    def __init__(self, c: float = 0.5):
        assert 0.0 <= c <= 1.0
        self.c = c
    def sample_for_all_agents(self, n: int, rng: RNG) -> np.ndarray:
        return np.full(n, self.c)

class TriangularItem(ItemDist):
    """Triangular on [0,1] with mode m in [0,1]."""
    def __init__(self, mode: float = 0.5):
        assert 0.0 <= mode <= 1.0
        self.mode = mode
    def sample_for_all_agents(self, n: int, rng: RNG) -> np.ndarray:
        u = rng.uniform(0.0, 1.0, size=n)
        m = self.mode
        left = (u < m)
        x = np.empty(n, dtype=float)
        # inverse CDF pieces on [0,1]
        x[left] = np.sqrt(u[left] * m)
        x[~left] = 1.0 - np.sqrt((1.0 - u[~left]) * (1.0 - m))
        return x

class BatesItem(ItemDist):
    """Bates(k): mean of k i.i.d. Uniform(0,1). Smooth, bell-ish on [0,1]."""
    def __init__(self, k: int = 4):
        assert k >= 1
        self.k = k
    def sample_for_all_agents(self, n: int, rng: RNG) -> np.ndarray:
        u = rng._g.uniform(0.0, 1.0, size=(n, self.k))
        return u.mean(axis=1)

class LogitNormalItem(ItemDist):
    """
    Logit-Normal: X = logistic(Z), Z ~ N(mu, sigma).
    Support strictly in (0,1). (No clipping.)
    """
    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        assert sigma > 0
        self.mu, self.sigma = mu, sigma
    def sample_for_all_agents(self, n: int, rng: RNG) -> np.ndarray:
        z = rng._g.normal(self.mu, self.sigma, size=n)
        return expit(z)  # numerically stable sigmoid

class TruncatedExponential01Item(ItemDist):
    """
    Exponential(rate=lambda) truncated to [0,1] with proper renormalization.
    Inverse CDF on [0,1]: x = -ln(1 - u*(1 - e^{-λ})) / λ
    """
    def __init__(self, lam: float = 3.0):
        assert lam > 0
        self.lam = lam
    def sample_for_all_agents(self, n: int, rng: RNG) -> np.ndarray:
        u = rng.uniform(0.0, 1.0, size=n)
        return -np.log(1.0 - u * (1.0 - np.exp(-self.lam))) / self.lam

class JohnsonSBItem(ItemDist):
    """
    Johnson SB(a, b) on (0,1). SciPy parameterization matches support (0,1).
    'a' controls tailweight, 'b' controls shape.
    """
    def __init__(self, a: float = 1.0, b: float = 1.0):
        assert a > 0 and b > 0
        self.a, self.b = a, b
    def sample_for_all_agents(self, n: int, rng: RNG) -> np.ndarray:
        return johnsonsb.rvs(self.a, self.b, size=n, random_state=rng._g)

class TruncatedLogistic01Item(ItemDist):
    """
    Logistic(loc=mu, scale=s) truncated to [0,1], exact inverse-CDF sampling.
    CDF: F(x)=expit((x-mu)/s). For truncation, invert on [F(0), F(1)].
    """
    def __init__(self, mu: float = 0.5, s: float = 0.1):
        assert s > 0
        self.mu, self.s = mu, s
    def sample_for_all_agents(self, n: int, rng: RNG) -> np.ndarray:
        F0 = expit((0.0 - self.mu)/self.s)
        F1 = expit((1.0 - self.mu)/self.s)
        u = rng.uniform(0.0, 1.0, size=n)
        p = F0 + u * (F1 - F0)
        return self.mu + self.s * logit(p)
