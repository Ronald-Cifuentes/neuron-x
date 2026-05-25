#!/usr/bin/env python3
"""
nx-1 1.0
LDNC — Living Digital Neural Cell

Strict implementation of the formal architectural document.

B1: Active boundary          — complete state, dynamic permeability
B2: Metabolism               — 5-stage chain, no perfect conversion
B3: Homeostasis              — regulatory network with error signals (no hardcoded ifs)
B4: Cognitive substrate      — recurrent network + STDP + degradation
B5: Material memory          — 4 causal subtypes: structural/regulatory/adaptive/heritable
B6: Repair                   — repairs ALL blocks, competes with reproduction
B7: Reproduction             — 4 inheritance types, possible failure, development as process
B8: Ecological coupling      — spatial world, neurally guided action
B9: Identity                 — computed variable I, organizational death M3

S(t) = {R_ext, R_int, A, M, P, W, X, G, T, C, I}
Death: M1 metabolic | M2 structural | M3 organizational
Degradation: D1-D7
Inheritance: H1-H4
"""

import argparse, math, random, time, json, threading, uuid, copy
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

# ─────────────────────────────────────────────────────────────
# SIMULATION SCALE
# ─────────────────────────────────────────────────────────────
#
# Change MAX_CELLS and the rest of the ecological constants needed will be
# derived from it. The central rule is non-negotiable: if there is a maximum
# occupancy of one cell per coordinate, the world must have at least
# max_cells habitable coordinates.
MAX_CELLS = 10_000
BASE_MAX_CELLS = 10_000
BASE_SOURCE_COUNT = 320
BASE_PRIMORDIAL_CELLS = 64
BASE_PREWARM_TICKS = 90
MIN_WORLD_SIDE = 12
MAX_DENSE_WORLD_CELLS = 25_000_000

# ─────────────────────────────────────────────────────────────
# EVOLUTIONARY ENGINE CONSTANTS
# ─────────────────────────────────────────────────────────────
N_COLONIES = 16
TOURNAMENT_INTERVAL = 2000   # ticks between selection events
MIGRATION_INTERVAL: int = 500          # ticks between inter-colony emigration pulses
GLOBAL_SIGNAL_LEAKAGE: float = 0.04   # fraction of cross-colony mean signal injected per tick
EXCIT_THRESHOLD = 0.55       # spike threshold for discrete firing
REFRACTORY_PERIOD = 4        # ticks of silence after a spike
APPLICATION_JSON = 'application/json'
NP_RNG = np.random.default_rng(0)

# ── Speed control ──────────────────────────────────────────────────────────
# Each label maps to the inter-tick sleep in seconds.
# real-time  = 1 tick per wall-clock second (the reference unit).
# accelerate = more ticks per second   → shorter sleep.
# decelerate = fewer ticks per second  → longer sleep.
SPEED_INTERVALS: Dict[str, float] = {
    "real-time":            1.0,
    "accelerate=2x":        0.5,
    "accelerate=3x":        1.0 / 3.0,
    "accelerate=10x":       0.1,
    "accelerate=100x":      0.01,
    "accelerate=1000x":     0.001,
    "accelerate=10000x":    0.0001,
    "accelerate=100000x":   0.00001,
    "accelerate=1000000x":  0.000001,
    "decelerate=2x":        2.0,
    "decelerate=3x":        3.0,
    "decelerate=10x":       10.0,
    "decelerate=100x":      100.0,
    "decelerate=1000x":     1000.0,
    "decelerate=10000x":    10000.0,
    "decelerate=100000x":   100000.0,
    "decelerate=1000000x":  1000000.0,
}


def speed_to_interval(speed: str) -> float:
    """Return the inter-tick sleep (seconds) for a given speed label."""
    return SPEED_INTERVALS[speed]


@dataclass(frozen=True)
class SimulationScale:
    max_cells: int
    world_width: int
    world_height: int
    n_sources: int
    primordial_cells: int
    prewarm_ticks: int

    @property
    def world_capacity(self) -> int:
        return self.world_width * self.world_height


def derive_simulation_scale(max_cells: int) -> SimulationScale:
    """
    Derives the spatial and trophic variables that depend on max_cells.

    This simulation uses a dense world with one cell per coordinate. Therefore
    the correct minimum world size is ceil(sqrt(max_cells))². For very large
    values, the current dense architecture is no longer memory-executable; in
    that case we fail early rather than lying with an impossible colony.
    """
    max_cells = int(max_cells)
    if max_cells < 1:
        raise ValueError("MAX_CELLS must be >= 1")

    side = max(MIN_WORLD_SIDE, math.ceil(math.sqrt(max_cells)))
    capacity = side * side
    if capacity < max_cells:
        raise AssertionError("Derived spatial capacity does not cover MAX_CELLS")
    if capacity > MAX_DENSE_WORLD_CELLS:
        raise ValueError(
            "MAX_CELLS requires a dense world of "
            f"{capacity:,} positions. The current dense-matrix architecture "
            "cannot guarantee survival at that scale without changing the spatial model."
        )

    source_ratio = BASE_SOURCE_COUNT / BASE_MAX_CELLS
    n_sources = min(capacity, max(6, math.ceil(max_cells * source_ratio)))
    primordial_cells = max(1, min(max_cells, BASE_PRIMORDIAL_CELLS))
    prewarm_ticks = max(BASE_PREWARM_TICKS, side)

    return SimulationScale(
        max_cells=max_cells,
        world_width=side,
        world_height=side,
        n_sources=n_sources,
        primordial_cells=primordial_cells,
        prewarm_ticks=prewarm_ticks,
    )

# ─────────────────────────────────────────────────────────────
# BIOSEMIOTIC INTERCELLULAR COMMUNICATION CHANNELS
# ─────────────────────────────────────────────────────────────
#
# These are not "human messages" or ML embeddings. They are causal, costly,
# diffusive, and degradable fields: digital equivalents of local chemical signals.
# Each channel is anchored to a vital variable of the cell, so that its
# meaning is not arbitrary but metabolic/homeostatic.
SIGNAL_CHANNELS = (
    "nutrient_beacon",      # exploitable resource here / path toward resource
    "toxin_alarm",         # external danger or local toxic load
    "energy_need",         # internal energy deficit
    "repair_need",         # damage/boundary/memory requires repair
    "reproduction_ready",  # mature/stable cell, lineage ecological signal
    "crowding",            # too much local density, avoid spatial saturation
    "death_trace",         # necrosignal: nearby death/organizational collapse
)
SIGNAL_IDX = {name: i for i, name in enumerate(SIGNAL_CHANNELS)}
N_SIGNAL_CHANNELS = len(SIGNAL_CHANNELS)

# ─────────────────────────────────────────────────────────────
# SPATIAL WORLD (B8 substrate)
# ─────────────────────────────────────────────────────────────

class SpatialWorld:
    """2D world with discrete diffusion, nutrient sources, and toxin accumulation."""

    def __init__(self, width: int = 40, height: int = 40,
                 n_sources: int = 6, rng: random.Random = None,
                 source_strength: float = 3.5,
                 perturbation_interval: int = 0,
                 perturbation_strength: float = 1.0):
        self.W = width
        self.H = height
        self.rng = rng or random.Random()
        self.source_strength = source_strength
        # grids
        self.nutrients = np.zeros((height, width), dtype=np.float64)
        self.toxins    = np.zeros((height, width), dtype=np.float64)
        # Intercellular communication fields: C × H × W.
        # Vectorized to keep cost low: O(channels × world), not O(cells²).
        self.signals   = np.zeros((N_SIGNAL_CHANNELS, height, width), dtype=np.float64)
        # Morphogenetic gradients: fixed positional fields for differentiating
        # space and enabling evolution of heritable body plans.
        # morphogen_a: anterior-posterior axis (x=0 → 1.0, x=W-1 → 0.0)
        # morphogen_b: dorsal-ventral axis     (y=0 → 1.0, y=H-1 → 0.0)
        self.morphogen_a = np.zeros((height, width), dtype=np.float64)
        self.morphogen_b = np.zeros((height, width), dtype=np.float64)
        self.morphogen_a[:, :] = np.linspace(1.0, 0.0, width)[np.newaxis, :]
        self.morphogen_b[:, :] = np.linspace(1.0, 0.0, height)[:, np.newaxis]
        self.occupied  = {}  # (x,y) -> cell_id
        # fixed nutrient sources
        self.sources = [(self.rng.randint(2, width-3), self.rng.randint(2, height-3))
                        for _ in range(n_sources)]
        # initial seed
        for sx, sy in self.sources:
            self.nutrients[sy, sx] = 80.0
        self.tick_count = 0
        # periodic environmental perturbation (Feature E)
        self.perturbation_interval = perturbation_interval
        self.perturbation_strength = perturbation_strength

    def tick(self):
        """Discrete diffusion + source emission + evaporation."""
        # nutrient diffusion
        lap = (
            np.roll(self.nutrients, 1, 0) + np.roll(self.nutrients, -1, 0) +
            np.roll(self.nutrients, 1, 1) + np.roll(self.nutrients, -1, 1) -
            4 * self.nutrients
        )
        self.nutrients += 0.08 * lap
        # toxin diffusion
        lap_t = (
            np.roll(self.toxins, 1, 0) + np.roll(self.toxins, -1, 0) +
            np.roll(self.toxins, 1, 1) + np.roll(self.toxins, -1, 1) -
            4 * self.toxins
        )
        self.toxins += 0.05 * lap_t

        # biosemiotic signal diffusion.
        # Signals = fast, local, degradable. Evaporation prevents infinite memory
        # and forces communication situated in the ecological present.
        for ch in range(N_SIGNAL_CHANNELS):
            grid = self.signals[ch]
            lap_s = (
                np.roll(grid, 1, 0) + np.roll(grid, -1, 0) +
                np.roll(grid, 1, 1) + np.roll(grid, -1, 1) -
                4 * grid
            )
            # alarms and death travel slightly faster than cooperative signals
            diff = 0.14 if ch in (SIGNAL_IDX["toxin_alarm"], SIGNAL_IDX["death_trace"]) else 0.09
            decay = 0.925 if ch in (SIGNAL_IDX["energy_need"], SIGNAL_IDX["repair_need"]) else 0.945
            self.signals[ch] += diff * lap_s
            self.signals[ch] *= decay

        # source emission
        for sx, sy in self.sources:
            self.nutrients[sy, sx] = min(120.0, self.nutrients[sy, sx] + self.source_strength)
        # gentle evaporation
        self.nutrients *= 0.995
        self.toxins    *= 0.990
        np.clip(self.nutrients, 0, 120, out=self.nutrients)
        np.clip(self.toxins,    0,  60, out=self.toxins)
        np.clip(self.signals,   0, 100, out=self.signals)

        # Morphogenetic diffusion (slow, α=0.04) + re-imposition of boundary sources.
        # Borders act as fixed sources/sinks → stable gradient but
        # not infinitely rigid; cells can perturb locally.
        for arr in (self.morphogen_a, self.morphogen_b):
            lap = (np.roll(arr, 1, 0) + np.roll(arr, -1, 0) +
                   np.roll(arr, 1, 1) + np.roll(arr, -1, 1) - 4 * arr)
            arr += 0.04 * lap
        self.morphogen_a[:, 0]  = 1.0   # anterior source
        self.morphogen_a[:, -1] = 0.0   # posterior sink
        self.morphogen_b[0, :]  = 1.0   # dorsal source
        self.morphogen_b[-1, :] = 0.0   # ventral sink
        np.clip(self.morphogen_a, 0.0, 1.0, out=self.morphogen_a)
        np.clip(self.morphogen_b, 0.0, 1.0, out=self.morphogen_b)

        # ── Periodic environmental perturbation (Feature E)
        if (self.perturbation_interval > 0 and self.tick_count > 0 and
                self.tick_count % self.perturbation_interval == 0):
            px = self.rng.randint(0, self.W - 1)
            py = self.rng.randint(0, self.H - 1)
            self.deposit_toxin(px, py, self.perturbation_strength)
            if self.sources:
                idx = self.rng.randint(0, len(self.sources) - 1)
                sx, sy = self.sources[idx]
                sx = (sx + self.rng.randint(-1, 1)) % self.W
                sy = (sy + self.rng.randint(-1, 1)) % self.H
                self.sources[idx] = (sx, sy)

        self.tick_count += 1

    def sample(self, x: int, y: int) -> Tuple[float, float]:
        x = x % self.W; y = y % self.H
        return float(self.nutrients[y, x]), float(self.toxins[y, x])

    def sample_signals(self, x: int, y: int) -> np.ndarray:
        """Local intercellular signal vector at (x,y)."""
        x = x % self.W; y = y % self.H
        return self.signals[:, y, x].copy()

    def sample_morphogens(self, x: int, y: int) -> Tuple[float, float]:
        """Returns (morphogen_a, morphogen_b) at (x,y) — positional information."""
        x = x % self.W; y = y % self.H
        return float(self.morphogen_a[y, x]), float(self.morphogen_b[y, x])

    def deposit_signal(self, x: int, y: int, channel, amount: float):
        """Deposits a local biosemiotic signal. Channel can be str or index."""
        if amount <= 0:
            return
        idx = SIGNAL_IDX[channel] if isinstance(channel, str) else int(channel)
        x = x % self.W; y = y % self.H
        self.signals[idx, y, x] = min(100.0, self.signals[idx, y, x] + float(amount))

    def signal_gradient(self, x: int, y: int, channel) -> Tuple[float, float]:
        """Gradiente espacial de un canal comunicativo."""
        idx = SIGNAL_IDX[channel] if isinstance(channel, str) else int(channel)
        return self.gradient(x, y, self.signals[idx])

    def gradient(self, x: int, y: int, grid: np.ndarray) -> Tuple[float, float]:
        """Centered gradient at (x,y)."""
        x = x % self.W; y = y % self.H
        dx = (grid[y, (x+1)%self.W] - grid[y, (x-1)%self.W]) / 2.0
        dy = (grid[(y+1)%self.H, x] - grid[(y-1)%self.H, x]) / 2.0
        return float(dx), float(dy)

    def consume_nutrient(self, x: int, y: int, amount: float) -> float:
        x = x % self.W; y = y % self.H
        actual = min(self.nutrients[y, x], amount)
        self.nutrients[y, x] -= actual
        return actual

    def deposit_toxin(self, x: int, y: int, amount: float):
        x = x % self.W; y = y % self.H
        self.toxins[y, x] = min(60.0, self.toxins[y, x] + amount)

    def deposit_nutrient(self, x: int, y: int, amount: float):
        x = x % self.W; y = y % self.H
        self.nutrients[y, x] = min(120.0, self.nutrients[y, x] + amount)

    def is_occupied(self, x: int, y: int) -> bool:
        return (x % self.W, y % self.H) in self.occupied

    def get_neighbor_cells(self, x: int, y: int, radius: int,
                           cells: Dict[str, "Cell"]) -> List[Dict]:
        """
        Local cellular neighborhood on the spatial torus.

        Returns enough biological context for adhesion, differentiation,
        and surveillance rules without introducing O(cells²) search.
        """
        x = x % self.W; y = y % self.H
        radius = max(1, int(radius))
        seen: Set[str] = set()
        return [
            neighbor for dx, dy, nx, ny in self._neighbor_positions(x, y, radius)
            if (neighbor := self._neighbor_snapshot(dx, dy, nx, ny, cells, seen)) is not None
        ]

    def _neighbor_positions(self, x: int, y: int, radius: int):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if (dx, dy) != (0, 0) and dx * dx + dy * dy <= radius * radius:
                    yield dx, dy, (x + dx) % self.W, (y + dy) % self.H

    def _neighbor_snapshot(self, dx: int, dy: int, nx: int, ny: int,
                           cells: Dict[str, "Cell"], seen: Set[str]) -> Optional[Dict]:
        cid = self.occupied.get((nx, ny))
        if cid is None or cid in seen:
            return None
        cell = cells.get(cid)
        if cell is None or not cell.alive:
            return None
        seen.add(cid)
        cell_type = getattr(cell, "cell_type", None)
        return {
            "id": cid,
            "x": nx,
            "y": ny,
            "dx": dx,
            "dy": dy,
            "distance": math.sqrt(dx * dx + dy * dy),
            "phase": cell.phase.value,
            "atp": cell.metabolism.atp_fraction,
            "damage": cell.damage_x,
            "identity": cell.identity.I,
            "generation": cell._generation,
            "cell_type": cell_type.value if hasattr(cell_type, "value") else str(getattr(cell, "cell_type", "unknown")),
        }

    def register(self, x: int, y: int, cell_id: str):
        self.occupied[(x % self.W, y % self.H)] = cell_id

    def unregister(self, x: int, y: int):
        self.occupied.pop((x % self.W, y % self.H), None)

    def snapshot(self) -> Dict:
        return {
            "nutrients":   self.nutrients.tolist(),
            "toxins":      self.toxins.tolist(),
            "signals":     self.signals.tolist(),
            "signal_channels": list(SIGNAL_CHANNELS),
            "morphogen_a": self.morphogen_a.tolist(),
            "morphogen_b": self.morphogen_b.tolist(),
            "occupied":    [[k[0], k[1]] for k in self.occupied.keys()],
            "sources":     self.sources,
            "W": self.W, "H": self.H
        }


# ─────────────────────────────────────────────────────────────
# GENOME — encodes H1, H2, H3, H4
# ─────────────────────────────────────────────────────────────

@dataclass
class Genome:
    """
    Encodes the four inheritance types from the formal document.

    H1: Structural inheritance   — boundary and metabolism properties
    H2: Regulatory inheritance   — homeostasis w_reg matrix
    H3: Cognitive inheritance    — neural weight predispositions
    H4: Developmental inheritance — maturation timings and costs
    """
    # H1: Structural inheritance
    membrane_strength:      float = 0.85
    transport_capacity:     float = 0.70
    metabolic_base_rate:    float = 0.60
    conversion_efficiency:  float = 0.55
    repair_capacity_base:   float = 0.50
    waste_tolerance:        float = 0.40
    structural_mass_cap:    float = 100.0
    repair_material_cap:    float = 60.0
    reproductive_mass_cap:  float = 80.0

    # H2: Regulatory inheritance — w_reg is a 4×5 matrix
    # (4 priorities × 5 error signals)
    # Stored as a flat list of 20 floats
    w_reg_flat: List[float] = field(default_factory=lambda: [
        # row 0: maintenance   [e_energy, e_damage, e_waste, e_boundary, e_memory]
         1.5,  0.5,  0.3,  1.2,  0.4,
        # row 1: repair
         0.3,  1.8,  0.5,  0.8,  0.6,
        # row 2: action
         0.8,  0.2,  0.3,  0.2,  0.5,
        # row 3: reproduction
         0.4,  0.1,  0.2,  0.2,  0.1,
    ])

    # H3: Cognitive inheritance — initial neural weight biases
    # Network: 6 inputs → 4 hidden → 3 outputs
    # Stored as predispositions (initialization scale)
    neural_w_ih_flat: List[float] = field(default_factory=lambda:
        [random.gauss(0, 0.3) for _ in range(6*4)])  # 6×4 = 24
    neural_w_hh_flat: List[float] = field(default_factory=lambda:
        [random.gauss(0, 0.2) for _ in range(4*4)])  # 4×4 = 16
    neural_w_ho_flat: List[float] = field(default_factory=lambda:
        [random.gauss(0, 0.3) for _ in range(4*3)])  # 4×3 = 12

    # H3: plasticity rate
    neural_plasticity:      float = 0.02
    neural_excitability:    float = 0.5

    # H4: Developmental inheritance
    development_ticks:      int   = 18
    maturation_cost_rate:   float = 0.18
    repr_min_age:           int   = 55
    repr_threshold_energy:  float = 0.25
    repr_threshold_damage:  float = 0.25
    fidelity:               float = 0.92
    motility:               float = 0.65
    chemotaxis_gain:        float = 0.8
    toxin_avoidance:        float = 0.7
    sensor_range:           int   = 3

    # Heritable communication layer: emission, reception and body translation.
    # This allows distinct lineages to develop compatible or partially
    # incompatible dialects without using human language or ML training.
    signal_emission_strength:    float = 0.55
    signal_receptor_sensitivity: float = 0.65
    signal_selectivity:          float = 0.55
    signal_cost_factor:          float = 0.025
    signal_receptor_flat: List[float] = field(default_factory=lambda:
        [random.gauss(0, 0.25) for _ in range(N_SIGNAL_CHANNELS * 4)])

    # Feature F: heritable target firing rate for synaptic homeostasis.
    # Controls the fraction of ticks the cell should spike. STDP alone is
    # unstable; this parameter drives a compensating multiplicative scale.
    target_firing_rate: float = 0.15

    # Heritable morphogenetics: response curves to positional gradients.
    # 8 floats = 2 morphogens × 4 cell types [BOUNDARY, NEURON, METABOLIC, REPAIR].
    # Indices 0-3: response to morphogen A; 4-7: response to morphogen B.
    # Positive → biases toward that type; negative → inhibits it.
    morphogen_response_flat: List[float] = field(default_factory=lambda:
        [random.gauss(0.0, 0.25) for _ in range(8)])

    def morphogen_response(self) -> np.ndarray:
        return np.array(self.morphogen_response_flat, dtype=np.float64).reshape(2, 4)

    def w_reg(self) -> np.ndarray:
        return np.array(self.w_reg_flat, dtype=np.float64).reshape(4, 5)

    def neural_w_ih(self) -> np.ndarray:
        return np.array(self.neural_w_ih_flat, dtype=np.float64).reshape(4, 6)

    def neural_w_hh(self) -> np.ndarray:
        return np.array(self.neural_w_hh_flat, dtype=np.float64).reshape(4, 4)

    def neural_w_ho(self) -> np.ndarray:
        return np.array(self.neural_w_ho_flat, dtype=np.float64).reshape(3, 4)

    def signal_receptor_w(self) -> np.ndarray:
        return np.array(self.signal_receptor_flat, dtype=np.float64).reshape(4, N_SIGNAL_CHANNELS)

    @staticmethod
    def create(rng: random.Random) -> "Genome":
        g = Genome()
        # H1
        g.membrane_strength     = rng.uniform(0.6, 1.0)
        g.transport_capacity    = rng.uniform(0.4, 0.9)
        g.metabolic_base_rate   = rng.uniform(0.4, 0.8)
        g.conversion_efficiency = rng.uniform(0.35, 0.70)
        g.repair_capacity_base  = rng.uniform(0.3, 0.7)
        g.waste_tolerance       = rng.uniform(0.2, 0.6)
        # H2
        g.w_reg_flat = [rng.gauss(0.5, 0.4) for _ in range(20)]
        # H3
        g.neural_w_ih_flat = [rng.gauss(0, 0.4) for _ in range(24)]
        g.neural_w_hh_flat = [rng.gauss(0, 0.25) for _ in range(16)]
        g.neural_w_ho_flat = [rng.gauss(0, 0.35) for _ in range(12)]
        g.neural_plasticity   = rng.uniform(0.005, 0.05)
        g.neural_excitability = rng.uniform(0.3, 0.8)
        # H4
        g.development_ticks       = rng.randint(8, 20)
        g.maturation_cost_rate    = rng.uniform(0.10, 0.22)
        g.repr_min_age            = rng.randint(35, 75)
        g.repr_threshold_energy   = rng.uniform(0.14, 0.30)
        g.repr_threshold_damage   = rng.uniform(0.15, 0.40)
        g.fidelity                = rng.uniform(0.80, 0.98)
        g.motility                = rng.uniform(0.3, 0.9)
        g.chemotaxis_gain         = rng.uniform(0.4, 1.2)
        g.toxin_avoidance         = rng.uniform(0.4, 1.0)
        g.sensor_range            = rng.randint(2, 5)
        g.signal_emission_strength    = rng.uniform(0.25, 0.95)
        g.signal_receptor_sensitivity = rng.uniform(0.25, 0.95)
        g.signal_selectivity          = rng.uniform(0.20, 0.95)
        g.signal_cost_factor          = rng.uniform(0.010, 0.050)
        g.signal_receptor_flat = [rng.gauss(0, 0.35) for _ in range(N_SIGNAL_CHANNELS * 4)]
        g.morphogen_response_flat = [rng.gauss(0.0, 0.35) for _ in range(8)]
        g.target_firing_rate = rng.uniform(0.05, 0.40)
        return g

    def mutate(self, rng: random.Random) -> "Genome":
        child = copy.deepcopy(self)
        fid = self.fidelity

        def mf(v, lo, hi, sigma=0.05):
            if rng.random() > fid:
                v += rng.gauss(0, sigma * (hi - lo))
                if rng.random() < 0.10:  # occasional large jump
                    v += rng.gauss(0, sigma * (hi - lo) * 3)
            return max(lo, min(hi, v))

        def ml(lst, sigma=0.08):
            return [v + rng.gauss(0, sigma) if rng.random() > fid else v for v in lst]

        # H1
        child.membrane_strength     = mf(child.membrane_strength,     0.1, 1.0)
        child.transport_capacity    = mf(child.transport_capacity,     0.1, 1.0)
        child.metabolic_base_rate   = mf(child.metabolic_base_rate,    0.1, 1.0)
        child.conversion_efficiency = mf(child.conversion_efficiency,  0.1, 0.85)
        child.repair_capacity_base  = mf(child.repair_capacity_base,   0.1, 1.0)
        child.waste_tolerance       = mf(child.waste_tolerance,        0.1, 0.8)
        # H2
        child.w_reg_flat = ml(child.w_reg_flat, 0.12)
        # H3
        child.neural_w_ih_flat    = ml(child.neural_w_ih_flat, 0.08)
        child.neural_w_hh_flat    = ml(child.neural_w_hh_flat, 0.06)
        child.neural_w_ho_flat    = ml(child.neural_w_ho_flat, 0.08)
        child.neural_plasticity   = mf(child.neural_plasticity,   0.001, 0.1, 0.02)
        child.neural_excitability = mf(child.neural_excitability, 0.1,   1.0, 0.05)
        # H4
        child.development_ticks     = max(5,  int(mf(child.development_ticks,     5,  30, 0.1)))
        child.maturation_cost_rate  = mf(child.maturation_cost_rate, 0.08, 0.30, 0.05)
        child.repr_min_age          = max(20, int(mf(child.repr_min_age,          20, 120, 0.1)))
        child.repr_threshold_energy = mf(child.repr_threshold_energy, 0.10, 0.55)
        child.repr_threshold_damage = mf(child.repr_threshold_damage, 0.05, 0.6)
        child.fidelity              = mf(child.fidelity,              0.5, 0.999, 0.02)
        child.motility              = mf(child.motility,              0.1, 1.0)
        child.chemotaxis_gain       = mf(child.chemotaxis_gain,       0.1, 2.0)
        child.toxin_avoidance       = mf(child.toxin_avoidance,       0.1, 2.0)
        child.sensor_range          = max(1, min(6, int(mf(child.sensor_range, 1, 6, 0.1))))
        child.signal_emission_strength    = mf(child.signal_emission_strength,    0.05, 1.5)
        child.signal_receptor_sensitivity = mf(child.signal_receptor_sensitivity, 0.05, 1.5)
        child.signal_selectivity          = mf(child.signal_selectivity,          0.05, 1.5)
        child.signal_cost_factor          = mf(child.signal_cost_factor,          0.001, 0.08)
        child.signal_receptor_flat        = ml(child.signal_receptor_flat, 0.07)
        child.morphogen_response_flat     = ml(child.morphogen_response_flat, 0.06)
        child.target_firing_rate          = mf(child.target_firing_rate, 0.02, 0.50, 0.03)
        return child

    def recombine(self, other: "Genome", rng: random.Random) -> "Genome":
        """Sexual recombination: element-wise mixing of two genomes (Feature C).

        Returns an unmutated offspring — mutation is applied separately in
        build_offspring_genome / mutate(), preserving the fidelity semantics.
        """
        child = copy.deepcopy(self)
        scalar_attrs = [
            'membrane_strength', 'transport_capacity', 'metabolic_base_rate',
            'conversion_efficiency', 'repair_capacity_base', 'waste_tolerance',
            'structural_mass_cap', 'repair_material_cap', 'reproductive_mass_cap',
            'neural_plasticity', 'neural_excitability', 'development_ticks',
            'maturation_cost_rate', 'repr_min_age', 'repr_threshold_energy',
            'repr_threshold_damage', 'fidelity', 'motility', 'chemotaxis_gain',
            'toxin_avoidance', 'sensor_range', 'signal_emission_strength',
            'signal_receptor_sensitivity', 'signal_selectivity', 'signal_cost_factor',
            'target_firing_rate',
        ]
        for attr in scalar_attrs:
            if rng.random() < 0.5:
                setattr(child, attr, getattr(other, attr))
        list_attrs = [
            'w_reg_flat', 'neural_w_ih_flat', 'neural_w_hh_flat', 'neural_w_ho_flat',
            'signal_receptor_flat', 'morphogen_response_flat',
        ]
        for attr in list_attrs:
            a_vals = getattr(self, attr)
            b_vals = getattr(other, attr)
            setattr(child, attr, [
                bv if rng.random() < 0.5 else av
                for av, bv in zip(a_vals, b_vals)
            ])
        return child


# ─────────────────────────────────────────────────────────────
# B1: ACTIVE BOUNDARY
# ─────────────────────────────────────────────────────────────

@dataclass
class Boundary:
    """
    B1: Active boundary.
    Filters inputs, regulates outputs, protects interior, sustains individuation.
    """
    c_integrity:            float = 1.0   # total integrity [0,1]
    c_permeability_resource: float = 0.7  # how much resource passes through
    c_permeability_signal:  float = 0.8   # signals
    c_permeability_toxin:   float = 0.15  # toxins that enter (low = good)
    c_transport_capacity:   float = 0.7   # active transport capacity
    c_maintenance_cost:     float = 0.05  # ATP per tick to maintain
    c_permanent_damage:     float = 0.0   # irrecoverable damage
    c_neural_gate:          float = 1.0   # modulation from B4 [0,1]

    def degrade(self, tox_external: float, tox_internal: float,
                age_ticks: int, genome: "Genome") -> float:
        """
        D1: Structural boundary degradation.
        Returns damage inflicted this tick.
        """
        # base damage from external toxins (reduced by current integrity)
        dmg_ext = tox_external * (1.0 - self.c_integrity * 0.5) * 0.002
        # damage from internal toxins (waste)
        dmg_int = tox_internal * 0.0006
        # age wear
        dmg_age = age_ticks * 0.000006
        # total damage attenuated by genomic membrane_strength
        total_dmg = (dmg_ext + dmg_int + dmg_age) / (genome.membrane_strength + 0.1)

        # apply damage
        self.c_integrity = max(0.0, self.c_integrity - total_dmg)
        # permanent fraction of accumulated damage (irrecoverable)
        self.c_permanent_damage = min(0.6, self.c_permanent_damage + total_dmg * 0.08)

        # permeability deteriorates with integrity
        self.c_permeability_resource = 0.5 + 0.5 * self.c_integrity * genome.transport_capacity
        self.c_permeability_toxin    = 0.05 + 0.3 * (1.0 - self.c_integrity)

        return total_dmg

    def repair(self, atp: float, structural: float,
               repair_fraction: float, genome: "Genome") -> Tuple[float, float]:
        """
        Repairs boundary using ATP and structural mass.
        repair_fraction ∈ [0,1]: portion of repair budget assigned to this block.
        Returns (atp_consumed, structural_consumed).
        """
        if atp <= 0 or structural <= 0:
            return 0.0, 0.0

        max_repair = genome.repair_capacity_base * repair_fraction * 0.12
        recovery_target = min(max_repair, 1.0 - self.c_integrity)
        if recovery_target <= 0:
            return 0.0, 0.0

        # cannot repair beyond the irrecoverable damage
        actual_repair = min(recovery_target, 1.0 - self.c_permanent_damage - self.c_integrity)
        actual_repair = max(0, actual_repair)

        atp_cost  = actual_repair * 4.0
        mass_cost = actual_repair * 2.0

        if atp_cost > atp or mass_cost > structural:
            scale = min(atp / (atp_cost + 1e-9), structural / (mass_cost + 1e-9))
            actual_repair *= scale
            atp_cost  *= scale
            mass_cost *= scale

        self.c_integrity = min(1.0 - self.c_permanent_damage, self.c_integrity + actual_repair)
        self.c_transport_capacity = 0.4 + 0.6 * self.c_integrity * genome.transport_capacity
        return atp_cost, mass_cost

    def apply_neural_gate(self, gate_signal: float):
        """B4 modulates selective boundary opening (F3 flow)."""
        self.c_neural_gate = max(0.2, min(1.0, gate_signal))
        # Modulation adjusts resource permeability
        self.c_permeability_resource *= (0.7 + 0.3 * self.c_neural_gate)
        self.c_permeability_resource  = min(1.0, self.c_permeability_resource)

    @property
    def is_dead(self) -> bool:
        return self.c_integrity < 0.08  # M2

    def to_dict(self) -> Dict:
        return {k: round(float(v), 4) for k, v in self.__dict__.items()}

@dataclass
class Metabolism:
    """
    B2: Metabolism.
    5-stage chain: raw_resource → intermediate → ATP → structural_mass → waste.
    No perfect conversion. There is always dissipation, waste, inefficiency.
    """
    r_raw:       float = 30.0   # internal raw resource
    r_int:       float = 0.0    # metabolic intermediate
    a_free:      float = 40.0   # ATP / free energy
    m_struct:    float = 20.0   # structural mass
    p_repair:    float = 10.0   # repair material
    q_repro:     float = 0.0    # reproductive material
    w_waste:     float = 0.0    # waste / internal toxic load
    eta_metabolic: float = 1.0  # metabolic efficiency [0,1] (degrades with D2)
    phi_dissipation: float = 0.05  # basal dissipation rate

    # caps
    r_raw_cap:  float = 120.0
    a_free_cap: float = 150.0
    m_struct_cap: float = 100.0
    p_repair_cap: float = 60.0
    q_repro_cap:  float = 80.0
    w_waste_cap:  float = 80.0

    # tracking for organizational closure
    atp_produced_last_tick: float = 0.0
    waste_produced_last_tick: float = 0.0

    def step(self, genome: "Genome", tox_internal: float,
             maintenance_priority: float) -> Dict:
        """
        Transforms resources in 5 stages.
        maintenance_priority ∈ [0,1] comes from B3.
        Returns tick flows.
        """
        # D2: efficiency degraded by waste and internal damage
        self.eta_metabolic = max(0.1, 1.0 - tox_internal * 0.015 - self.w_waste * 0.008)

        # ── STAGE 1: raw resource → intermediate
        # maintenance_priority modulates what fraction of metabolism is active,
        # but the base rate must be high enough to sustain the cell
        rate_e1 = genome.metabolic_base_rate * self.eta_metabolic * (0.5 + 0.5 * maintenance_priority)
        converted_to_int = min(self.r_raw, self.r_raw * rate_e1 * 0.8)
        self.r_raw -= converted_to_int
        self.r_int += converted_to_int * 0.88  # 12% lost as heat
        waste_e1    = converted_to_int * 0.04

        # ── STAGE 2: intermediate → ATP
        rate_e2 = genome.conversion_efficiency * self.eta_metabolic
        atp_from_int = min(self.r_int, self.r_int * rate_e2 * 0.90)
        self.r_int -= atp_from_int
        atp_produced = atp_from_int * 2.2  # energy conversion
        waste_e2 = atp_from_int * 0.07
        self.a_free = min(self.a_free_cap, self.a_free + atp_produced)
        self.atp_produced_last_tick = atp_produced

        # ── STAGE 3: ATP → structural mass
        # Maintenance priority directs the post-production fraction
        struct_alloc = 0.12 * maintenance_priority
        atp_for_struct = min(self.a_free * 0.3, self.a_free * struct_alloc)
        self.a_free -= atp_for_struct
        struct_produced = atp_for_struct * 0.65
        self.m_struct = min(self.m_struct_cap, self.m_struct + struct_produced)

        # ── STAGE 4: ATP → repair material
        repair_alloc = 0.08 * maintenance_priority
        atp_for_repair = min(self.a_free * 0.20, self.a_free * repair_alloc)
        self.a_free -= atp_for_repair
        repair_produced = atp_for_repair * 0.80
        self.p_repair = min(self.p_repair_cap, self.p_repair + repair_produced)

        # ── STAGE 5: surplus → reproductive material
        if self.a_free > self.a_free_cap * 0.45:
            atp_for_repro = self.a_free * 0.06
            self.a_free -= atp_for_repro
            repro_produced = atp_for_repro * 0.65
            self.q_repro = min(self.q_repro_cap, self.q_repro + repro_produced)
        else:
            repro_produced = 0.0

        # ── Total waste (there is always waste — no perfect conversion)
        total_waste = waste_e1 + waste_e2 + self.a_free * self.phi_dissipation * 0.005
        self.w_waste = min(self.w_waste_cap, self.w_waste + total_waste)
        self.waste_produced_last_tick = total_waste

        # ── Basal dissipation (cost of existing — reduced)
        self.a_free = max(0.0, self.a_free - self.a_free * self.phi_dissipation * 0.02)

        return {
            "atp_produced": atp_produced,
            "struct_produced": struct_produced,
            "repair_produced": repair_produced,
            "repro_produced": repro_produced,
            "waste_produced": total_waste,
            "eta": self.eta_metabolic
        }

    def consume_atp(self, amount: float) -> float:
        actual = min(self.a_free, amount)
        self.a_free -= actual
        return actual

    def consume_structural(self, amount: float) -> float:
        actual = min(self.m_struct, amount)
        self.m_struct -= actual
        return actual

    def consume_repair_material(self, amount: float) -> float:
        actual = min(self.p_repair, amount)
        self.p_repair -= actual
        return actual

    @property
    def atp_fraction(self) -> float:
        return self.a_free / (self.a_free_cap + 1e-9)

    @property
    def waste_fraction(self) -> float:
        return self.w_waste / (self.w_waste_cap + 1e-9)

    @property
    def is_starving(self) -> bool:
        return self.a_free < 3.0

    def repair_efficiency(self, repair_amount: float):
        """B6 restores metabolic efficiency (D2 reversal)."""
        self.eta_metabolic = min(1.0, self.eta_metabolic + repair_amount * 0.04)
        # clears internal waste
        waste_cleared = min(self.w_waste, repair_amount * 2.0)
        self.w_waste -= waste_cleared
        return waste_cleared

    def to_dict(self) -> Dict:
        return {k: round(float(v), 4) for k, v in self.__dict__.items()
                if not k.endswith('_cap')}


# ─────────────────────────────────────────────────────────────
# B3: HOMEOSTASIS / INTERNAL REGULATION
# ─────────────────────────────────────────────────────────────

class HomeostasisRegulator:
    """
    B3: Regulatory network.

    Does NOT use hardcoded ifs.
    Error signals → w_reg (heritable) → priority vector via softmax.
    Priorities assign ATP to: maintenance / repair / action / reproduction.

    G = {g_energy_error, g_damage_error, g_waste_error, g_boundary_error, g_memory_error,
         g_priority_maintenance, g_priority_repair, g_priority_action, g_priority_reproduction,
         g_stress, g_regulatory_coherence}
    """

    def __init__(self, genome: "Genome"):
        # w_reg: 4 outputs × 5 inputs — heritable (H2)
        self.w_reg = genome.w_reg().copy()
        # setpoints (can adapt via B5 regulatory memory)
        self.sp_energy   = 0.65   # desired ATP fraction
        self.sp_damage   = 0.20   # tolerable damage threshold
        self.sp_waste    = 0.30   # tolerable waste fraction
        self.sp_boundary = 0.75   # desired integrity
        self.sp_memory   = 0.70   # desired memory integrity

        # G state
        self.g_energy_error   = 0.0
        self.g_damage_error   = 0.0
        self.g_waste_error    = 0.0
        self.g_boundary_error = 0.0
        self.g_memory_error   = 0.0
        self.g_stress         = 0.0
        self.g_regulatory_coherence = 1.0

        # computed priorities (not hardcoded)
        self.p_maintenance  = 0.35
        self.p_repair       = 0.25
        self.p_action       = 0.25
        self.p_reproduction = 0.15

        # history for coherence
        self._priority_history = []
        self._error_history    = []

    def update(self, met: Metabolism, boundary: Boundary,
               damage: float, mem_integrity: float,
               mem_regulatory: np.ndarray) -> Dict:
        """
        Updates error signals and computes priorities via w_reg.
        mem_regulatory comes from B5 and adjusts setpoints (regulatory plasticity).
        """
        # ── Error signals (positive = problem)
        self.g_energy_error   = max(0, self.sp_energy   - met.atp_fraction)
        self.g_damage_error   = max(0, damage / 0.8     - self.sp_damage)
        self.g_waste_error    = max(0, met.waste_fraction - self.sp_waste)
        self.g_boundary_error = max(0, self.sp_boundary - boundary.c_integrity)
        self.g_memory_error   = max(0, self.sp_memory   - mem_integrity)

        e_vec = np.array([
            self.g_energy_error,
            self.g_damage_error,
            self.g_waste_error,
            self.g_boundary_error,
            self.g_memory_error
        ], dtype=np.float64)

        # ── Setpoint adjustment from regulatory memory (B5 → B3)
        if mem_regulatory is not None and len(mem_regulatory) == 5:
            # regulatory memory slightly biases the setpoints
            self.sp_energy   = np.clip(0.65 + mem_regulatory[0] * 0.05, 0.3, 0.9)
            self.sp_boundary = np.clip(0.75 + mem_regulatory[3] * 0.05, 0.4, 0.95)

        # ── Priorities via w_reg @ error_vector → softmax
        raw = self.w_reg @ e_vec
        raw = np.clip(raw, -8, 8)
        exp = np.exp(raw - raw.max())
        priorities = exp / (exp.sum() + 1e-9)

        self.p_maintenance  = float(priorities[0])
        self.p_repair       = float(priorities[1])
        self.p_action       = float(priorities[2])
        self.p_reproduction = float(priorities[3])

        # ── Systemic stress = RMS of errors
        self.g_stress = float(np.sqrt(np.mean(e_vec**2)))

        # ── Regulatory coherence: stability of the priority vector
        self._priority_history.append(priorities.copy())
        if len(self._priority_history) > 20:
            self._priority_history.pop(0)
        if len(self._priority_history) >= 5:
            arr = np.array(self._priority_history[-5:])
            # coherence = 1 - mean variance of priorities
            self.g_regulatory_coherence = float(
                max(0, 1.0 - np.mean(np.std(arr, axis=0)) * 4)
            )

        return {
            "priorities": {
                "maintenance": self.p_maintenance,
                "repair": self.p_repair,
                "action": self.p_action,
                "reproduction": self.p_reproduction
            },
            "stress": self.g_stress,
            "errors": {
                "energy": self.g_energy_error,
                "damage": self.g_damage_error,
                "waste": self.g_waste_error,
                "boundary": self.g_boundary_error,
                "memory": self.g_memory_error
            }
        }

    def adapt_w_reg(self, learning_signal: float):
        """
        Regulatory plasticity: slow adjustment of w_reg by memory signal (B5→B3).
        This allows the regulatory strategy to evolve within the cell's lifetime.
        """
        e_vec = np.array([
            self.g_energy_error, self.g_damage_error, self.g_waste_error,
            self.g_boundary_error, self.g_memory_error
        ])
        p_vec = np.array([self.p_maintenance, self.p_repair,
                          self.p_action, self.p_reproduction])
        # Hebb-like: reinforces connections that correlate with low stress
        if self.g_stress < 0.2:
            delta = np.outer(p_vec, e_vec) * learning_signal * 0.001
            self.w_reg += delta
            self.w_reg = np.clip(self.w_reg, -5, 5)

    def to_dict(self) -> Dict:
        return {
            "stress": round(self.g_stress, 4),
            "regulatory_coherence": round(self.g_regulatory_coherence, 4),
            "priorities": {
                "maintenance":  round(self.p_maintenance, 4),
                "repair":       round(self.p_repair, 4),
                "action":       round(self.p_action, 4),
                "reproduction": round(self.p_reproduction, 4)
            },
            "errors": {
                "energy":   round(self.g_energy_error, 4),
                "damage":   round(self.g_damage_error, 4),
                "waste":    round(self.g_waste_error, 4),
                "boundary": round(self.g_boundary_error, 4),
                "memory":   round(self.g_memory_error, 4)
            }
        }


# ─────────────────────────────────────────────────────────────
# B4: COGNITIVE-NEURAL SUBSTRATE
# ─────────────────────────────────────────────────────────────

class NeuralCore:
    """
    B4: Cognitive-neural substrate.
    Subsystem within the cell, not the cell itself.

    Architecture: 6 inputs → 4 hidden (recurrent) → 3 outputs
    Inputs:  [atp_frac, membrane_integrity, waste_norm, nut_local, tox_local, damage_norm]
    Outputs: [move_bias_x, move_bias_y, capture_modulation]

    + Internal prediction (predictive error t_prediction_error)
    + STDP-like: updates w_ih from predictive error
    + Degrades with D3 when energy is low or damage is high

    T = {t_internal_state, t_synaptic_matrix, t_excitation,
         t_prediction_error, t_adaptive_trace, t_action_bias}
    """

    def __init__(self, genome: "Genome"):
        # Synaptic weights (inherited predispositions H3)
        self.w_ih = genome.neural_w_ih().copy()  # 4×6
        self.w_hh = genome.neural_w_hh().copy()  # 4×4
        self.w_ho = genome.neural_w_ho().copy()  # 3×4

        # Prediction layer: predicts next input (6 outputs)
        self.w_pred = NP_RNG.normal(size=(6, 4)) * 0.2  # 6×4

        # Internal state
        self.t_internal_state   = np.zeros(4)   # h_t
        self.t_excitation       = 0.0
        self.t_prediction_error = 0.0
        self.t_adaptive_trace   = np.zeros(4)   # activation trace
        self.t_action_bias      = np.zeros(3)   # outputs

        # For D3: cognitive degradation
        self.t_noise_level      = 0.0
        self.t_integrity        = 1.0

        # Heritable parameters
        self.plasticity   = genome.neural_plasticity
        self.excitability = genome.neural_excitability

        # History for adaptive memory
        self._last_input    = np.zeros(6)
        self._input_history: List[np.ndarray] = []

        # Electrical GAP junction coupling (Feature A)
        self.gap_current_input: float = 0.0

        # Feature F: synaptic homeostasis — firing-rate regulation.
        # STDP alone is unstable; this multiplicative scale keeps mean
        # activity near the heritable target, preventing silence or saturation.
        self.target_firing_rate: float = genome.target_firing_rate
        self.firing_rate_trace: float = 0.0    # exponential avg of spike events
        self.homeostatic_scale: float = 1.0    # multiplied onto incoming synaptic current

    def step(self, inputs: np.ndarray, adaptive_trace_bias: np.ndarray,
             atp_available: float) -> Dict:
        """
        One tick of the neural substrate.

        inputs: [atp_frac, membrane_integrity, waste_norm, nut_local_norm, tox_local_norm, damage_norm]
        adaptive_trace_bias: bias from B5 adaptive memory
        atp_available: if low, degrades (D3)
        """
        # ── D3: cognitive degradation from energy shortage
        if atp_available < 5.0:
            self.t_noise_level = min(1.0, self.t_noise_level + 0.02)
            self.t_integrity = max(0.1, self.t_integrity - 0.005)
        else:
            self.t_noise_level = max(0.0, self.t_noise_level - 0.005)
            self.t_integrity = min(1.0, self.t_integrity + 0.002)

        # noise proportional to degradation (D3)
        noise = NP_RNG.normal(size=inputs.shape) * self.t_noise_level * 0.3

        # ── Predictive error (prediction of the previous input)
        if np.any(self._last_input != 0):
            predicted = np.tanh(self.w_pred @ self.t_internal_state)
            self.t_prediction_error = float(np.mean(np.abs(predicted - self._last_input)))
        else:
            self.t_prediction_error = 0.0

        # ── Recurrent step
        # adaptive memory bias (B5 → B4: previous activation biases)
        bias = adaptive_trace_bias[:4] if len(adaptive_trace_bias) >= 4 else np.zeros(4)
        noisy_input = inputs + noise
        h_new = np.tanh(
            self.w_ih @ noisy_input +
            self.w_hh @ self.t_internal_state * self.excitability +
            bias * 0.35 +
            self.gap_current_input  # GAP junction electrical coupling
        )
        self.clear_gap_current()  # consumed this tick
        self.t_internal_state = h_new

        # ── Outputs
        out = np.tanh(self.w_ho @ h_new)
        self.t_action_bias = out

        # ── Excitation = state norm
        self.t_excitation = float(np.mean(np.abs(h_new)))

        # ── Adaptive trace (decaying average)
        self.t_adaptive_trace = 0.92 * self.t_adaptive_trace + 0.08 * h_new

        # ── STDP-like: updates w_ih proportional to predictive error
        if self.t_prediction_error > 0.01 and atp_available > 10:
            delta_w = np.outer(h_new, noisy_input) * self.plasticity * self.t_prediction_error
            self.w_ih += delta_w
            # also updates w_pred
            self.w_pred += np.outer(noisy_input, h_new) * self.plasticity * 0.5
            # clamp weights
            self.w_ih   = np.clip(self.w_ih,   -3, 3)
            self.w_pred = np.clip(self.w_pred,  -3, 3)

        self._last_input = inputs.copy()
        self._input_history.append(inputs.copy())
        if len(self._input_history) > 50:
            self._input_history.pop(0)

        # ── Interpreted outputs
        move_x    = float(out[0])  # [-1, 1]
        move_y    = float(out[1])  # [-1, 1]
        capture_m = float((out[2] + 1) / 2)  # [0, 1] capture modulation

        return {
            "move_x": move_x,
            "move_y": move_y,
            "capture_modulation": capture_m,
            "excitation": self.t_excitation,
            "prediction_error": self.t_prediction_error,
            "integrity": self.t_integrity
        }

    def repair_weights(self, repair_amount: float):
        """B6 repairs neural substrate: reduces noise, restores integrity (D3 reversal)."""
        self.t_noise_level = max(0.0, self.t_noise_level - repair_amount * 0.1)
        self.t_integrity   = min(1.0, self.t_integrity   + repair_amount * 0.03)
        # small restoration of degraded weights
        self.w_ih   = np.clip(self.w_ih,   -3, 3)
        self.w_hh   = np.clip(self.w_hh,   -3, 3)
        self.w_ho   = np.clip(self.w_ho,   -3, 3)

    def get_heritable_predispositions(self) -> Tuple[List[float], List[float], List[float]]:
        """
        H3: Extracts predispositions for transmission to offspring.
        Does not transfer full learning — only tendencies.
        """
        # averages with original genomic weights to not transfer all learning
        w_ih_heir = (self.w_ih * 0.35).flatten().tolist()
        w_hh_heir = (self.w_hh * 0.35).flatten().tolist()
        w_ho_heir = (self.w_ho * 0.35).flatten().tolist()
        return w_ih_heir, w_hh_heir, w_ho_heir

    def update_homeostatic_scale(self, spiked: bool):
        """Feature F: multiplicative synaptic scaling for firing-rate homeostasis.

        Tracks an exponential average of spike events and adjusts homeostatic_scale
        so that the delivered synaptic current is scaled up when the cell fires too
        rarely and down when it fires too often. Stabilises STDP learning.
        """
        self.firing_rate_trace = (0.97 * self.firing_rate_trace +
                                  0.03 * (1.0 if spiked else 0.0))
        error = self.target_firing_rate - self.firing_rate_trace
        self.homeostatic_scale = float(
            np.clip(self.homeostatic_scale + 0.002 * error, 0.1, 4.0)
        )

    def apply_gap_current(self, amount: float):
        """Accumulates incoming electrical current from GAP junctions (Feature A)."""
        self.gap_current_input += amount

    def clear_gap_current(self):
        """Resets the accumulated GAP current (called in step())."""
        self.gap_current_input = 0.0

    def to_dict(self) -> Dict:
        return {
            "excitation":       round(self.t_excitation, 4),
            "prediction_error": round(self.t_prediction_error, 4),
            "integrity":           round(self.t_integrity, 4),
            "noise_level":         round(self.t_noise_level, 4),
            "action_bias":         [round(float(v), 3) for v in self.t_action_bias],
            "adaptive_trace":      [round(float(v), 3) for v in self.t_adaptive_trace],
            "homeostatic_scale":   round(self.homeostatic_scale, 4),
            "firing_rate_trace":   round(self.firing_rate_trace, 4),
        }


# ─────────────────────────────────────────────────────────────
# B5: MATERIAL MEMORY
# ─────────────────────────────────────────────────────────────

class MaterialMemory:
    """
    B5: Material memory.
    4 subtypes, all causally active, all degradable, all costly.

    h_structural:  boundary integrity history → feeds repair priority
    h_regulatory:  homeostatic error average  → biases B3 setpoints
    h_adaptive:    neural activation trace    → biases B4 inputs
    h_heritable:   stable heritable parameters → passed to offspring

    Mem = {h_structural, h_regulatory, h_adaptive, h_heritable, h_integrity, h_corruption_rate}
    """

    def __init__(self, genome: "Genome"):
        # Subtypes
        self.h_structural  = np.zeros(10)   # boundary integrity history (10 ticks)
        self.h_regulatory  = np.zeros(5)    # smoothed average of homeostatic errors
        self.h_adaptive    = np.zeros(4)    # neural activation trace (4 neurons)
        self.h_heritable   = np.array([     # stable heritable parameters
            genome.membrane_strength,
            genome.metabolic_base_rate,
            genome.conversion_efficiency,
            genome.repair_capacity_base,
            genome.waste_tolerance,
        ])

        # Global memory integrity
        self.h_integrity       = 1.0
        self.h_corruption_rate = 0.001

        # Maintenance cost per tick (reduced — memory is costly but not ruinous)
        self.maintenance_cost_per_tick = 0.08  # ATP

        self._struct_ptr = 0  # circular pointer for h_structural

    def update(self, boundary_integrity: float, homeostatic_errors: Dict[str, float],
               neural_trace: np.ndarray, atp_available: float,
               waste_internal: float, damage: float) -> Dict:
        """
        Updates the 4 memory subtypes.
        Costs ATP. Degrades if resources are unavailable (D4).
        """
        cost_paid = 0.0

        # ── D4: memory degradation from waste and lack of repair
        # Slow but real — memory has a material body
        corruption_increment = waste_internal * 0.00008 + damage * 0.0002
        self.h_corruption_rate = min(0.015, self.h_corruption_rate + corruption_increment)
        self.h_integrity = max(0.0, self.h_integrity - self.h_corruption_rate * 0.3)

        # If ATP available, pay maintenance and preserve memory
        if atp_available >= self.maintenance_cost_per_tick:
            cost_paid = self.maintenance_cost_per_tick
            # corruption decay if well maintained
            self.h_corruption_rate = max(0.0005, self.h_corruption_rate - 0.002)
            self.h_integrity = min(1.0, self.h_integrity + 0.005)
        else:
            # no maintenance → active corruption
            self.h_corruption_rate = min(0.015, self.h_corruption_rate + 0.001)

        # update scale by integrity
        update_scale = self.h_integrity

        # ── h_structural: circular update
        noise_struct = NP_RNG.normal() * self.h_corruption_rate * 0.1
        self.h_structural[self._struct_ptr] = boundary_integrity * update_scale + noise_struct
        self._struct_ptr = (self._struct_ptr + 1) % 10

        # ── h_regulatory: exponential average of homeostatic errors
        e_vec = np.array([
            homeostatic_errors.get('energy', 0),
            homeostatic_errors.get('damage', 0),
            homeostatic_errors.get('waste',  0),
            homeostatic_errors.get('boundary', 0),
            homeostatic_errors.get('memory', 0)
        ])
        noise_reg = NP_RNG.normal(size=5) * self.h_corruption_rate * 0.05
        self.h_regulatory = (0.95 * self.h_regulatory + 0.05 * e_vec * update_scale + noise_reg)
        self.h_regulatory = np.clip(self.h_regulatory, -1, 1)

        # ── h_adaptive: neural activation trace (B4 → B5)
        noise_adap = NP_RNG.normal(size=4) * self.h_corruption_rate * 0.05
        self.h_adaptive = (0.90 * self.h_adaptive + 0.10 * neural_trace * update_scale + noise_adap)

        # ── h_heritable: degrades very slowly (D4/D5)
        noise_her = NP_RNG.normal(size=5) * self.h_corruption_rate * 0.02
        self.h_heritable = np.clip(self.h_heritable + noise_her, 0.05, 1.5)

        return {
            "h_integrity": self.h_integrity,
            "corruption_rate": self.h_corruption_rate,
            "maintenance_cost": cost_paid
        }

    def repair_memory(self, repair_amount: float):
        """B6 repairs memory: reduces corruption, restores integrity (D4 reversal)."""
        self.h_corruption_rate = max(0.001, self.h_corruption_rate - repair_amount * 0.02)
        self.h_integrity       = min(1.0,   self.h_integrity        + repair_amount * 0.05)
        # small restoration of h_heritable
        # (partial repair, not perfect)
        self.h_heritable = np.clip(self.h_heritable, 0.05, 1.5)

    def get_structural_repair_signal(self) -> float:
        """
        B5 → B6: structural memory signal for repair priority.
        If history shows boundary deteriorating → urge more repair.
        """
        if np.all(self.h_structural == 0):
            return 0.5
        trend = np.mean(np.diff(self.h_structural)) if len(self.h_structural) > 1 else 0
        # negative trend → high repair signal
        return float(np.clip(0.5 - trend * 2.0, 0.0, 1.0))

    def get_regulatory_bias(self) -> np.ndarray:
        """B5 → B3: regulatory bias from memory."""
        return self.h_regulatory.copy()

    def get_adaptive_bias(self) -> np.ndarray:
        """B5 → B4: adaptive bias for cognition."""
        return self.h_adaptive.copy()

    def get_heritable_snapshot(self) -> np.ndarray:
        """H4/H1: current heritable parameters (for reproduction)."""
        return self.h_heritable.copy()

    def to_dict(self) -> Dict:
        return {
            "h_integrity":      round(self.h_integrity, 4),
            "corruption_rate":  round(self.h_corruption_rate, 4),
            "h_structural_mean": round(float(np.mean(self.h_structural)), 4),
            "h_regulatory":     [round(float(v), 3) for v in self.h_regulatory],
            "h_adaptive":       [round(float(v), 3) for v in self.h_adaptive],
            "h_heritable":      [round(float(v), 3) for v in self.h_heritable]
        }

class RepairSystem:
    """
    B6: Repair system.
    Repairs ALL blocks: boundary, metabolism, cognitive, memory.
    Consumes ATP and repair material.
    Coordinated by B3 priorities.

    Rep = {p_capacity, p_queue_damage, p_efficiency, p_latency}
    """

    def __init__(self, genome: "Genome"):
        self.p_capacity   = genome.repair_capacity_base
        self.p_efficiency = 0.8
        self.p_latency    = 2
        self.p_queue_damage = 0.0
        self._ticks_since_repair = 0
        self._atp_consumed_last = 0.0
        self._boundary_repaired_last = 0.0

    def execute(self, met: Metabolism, boundary: Boundary,
                neural: NeuralCore, memory: MaterialMemory,
                homeostasis: HomeostasisRegulator,
                damage: float, genome: "Genome") -> Tuple[float, Dict]:
        """
        Distributes repair budget among the 4 blocks.
        Returns (damage_reduced, repair_log).
        """
        self._ticks_since_repair += 1
        if self._ticks_since_repair < self.p_latency:
            return 0.0, {}

        self._ticks_since_repair = 0

        # Total budget: available ATP × repair priority
        atp_budget    = met.a_free * homeostasis.p_repair * 0.20
        struct_budget = met.p_repair * homeostasis.p_repair * 0.40

        if atp_budget < 0.5 or struct_budget < 0.5:
            return 0.0, {}

        atp_total    = met.consume_atp(atp_budget)
        struct_total = met.consume_repair_material(struct_budget)

        # Structural memory signal → adjusts fraction for boundary
        struct_repair_signal = memory.get_structural_repair_signal()

        # ── Repair distribution among blocks
        # Proportional to each block's damage + memory signal
        dmg_boundary = max(0, 1.0 - boundary.c_integrity)
        dmg_neural   = neural.t_noise_level
        dmg_memory   = 1.0 - memory.h_integrity
        dmg_metabolic = 1.0 - met.eta_metabolic

        total_need = dmg_boundary + dmg_neural + dmg_memory + dmg_metabolic + 1e-9
        # structural bias from B5
        boundary_extra = struct_repair_signal * 0.2

        frac_boundary = (dmg_boundary / total_need + boundary_extra)
        frac_neural   = dmg_neural   / total_need * (1 - boundary_extra)
        frac_memory   = dmg_memory   / total_need * (1 - boundary_extra)
        frac_metabolic= dmg_metabolic/ total_need * (1 - boundary_extra)

        # normalize
        s = frac_boundary + frac_neural + frac_memory + frac_metabolic + 1e-9
        frac_boundary /= s; frac_neural /= s
        frac_memory   /= s; frac_metabolic /= s

        # ── Repair each block
        atp_b, _ = boundary.repair(
            atp_total * frac_boundary, struct_total * frac_boundary,
            frac_boundary, genome
        )
        self._boundary_repaired_last = 1.0 - boundary.c_integrity

        neural.repair_weights(frac_neural * self.p_capacity * self.p_efficiency)
        memory.repair_memory(frac_memory * self.p_capacity * self.p_efficiency)
        waste_cleared = met.repair_efficiency(frac_metabolic * self.p_capacity * self.p_efficiency)

        total_atp_used = atp_b + (atp_total - atp_b) * 0.8
        self._atp_consumed_last = total_atp_used
        self.p_queue_damage = max(0, damage - self.p_capacity * 0.1)

        damage_reduced = self.p_capacity * self.p_efficiency * 0.08
        return damage_reduced, {
            "atp_used": total_atp_used,
            "frac_boundary": frac_boundary,
            "frac_neural": frac_neural,
            "frac_memory": frac_memory,
            "frac_metabolic": frac_metabolic,
            "waste_cleared": waste_cleared
        }

    def to_dict(self) -> Dict:
        return {
            "capacity":     round(self.p_capacity, 4),
            "efficiency":   round(self.p_efficiency, 4),
            "queue_damage": round(self.p_queue_damage, 4),
            "atp_last":     round(self._atp_consumed_last, 4)
        }


# ─────────────────────────────────────────────────────────────
# B7: REPRODUCTION AND DEVELOPMENT
# ─────────────────────────────────────────────────────────────

class ReproductionModule:
    """
    B7: Reproduction and development.

    Conditions (R1-R8): metabolic surplus, structural material,
    reproductive material, low damage, controlled waste, intact heritable memory,
    identity coherence, reproductive maturity.

    Inheritance H1-H4: structure, regulation, cognition, development.
    Possible failures: nonviable/malformed offspring, complete failure with cost.
    """

    def __init__(self, genome: "Genome"):
        self.r_maturity   = 0.0   # grows over time
        self.r_material   = 0.0   # accumulated reproductive material
        self.r_stability  = 1.0
        self.r_failure_risk = 0.0
        self.d_stage      = 0     # offspring development stage
        self.d_max_stage  = genome.development_ticks
        self._replicating = False
        self._abort_count = 0
        self._rng = random.Random()

    def tick_maturity(self, age: int, genome: "Genome", met: Metabolism,
                      damage: float):
        """Matures slowly. Requires minimum age (H4)."""
        if age >= genome.repr_min_age:
            self.r_maturity = min(1.0, self.r_maturity + 0.012)
        # accumulates reproductive material from metabolism
        self.r_material = min(genome.reproductive_mass_cap,
                              self.r_material + met.q_repro * 0.3)
        # failure risk grows with accumulated damage (D5)
        self.r_failure_risk = min(0.8, damage * 0.6 + (1 - met.eta_metabolic) * 0.3)

    def can_reproduce(self, met: Metabolism, memory: MaterialMemory,
                      damage: float, identity_value: float,
                      genome: "Genome") -> bool:
        """R1-R8: all conditions must be met."""
        if self._replicating:
            return False
        # R1: metabolic surplus
        if met.atp_fraction < genome.repr_threshold_energy:      return False
        # R2: structural material
        if met.m_struct < genome.structural_mass_cap * 0.40:     return False
        # R3: reproductive material
        if self.r_material < 15.0:                                return False
        # R4: low damage
        if damage > genome.repr_threshold_damage:                 return False
        # R5: controlled waste
        if met.waste_fraction > 0.45:                             return False
        # R6: intact heritable memory
        if memory.h_integrity < 0.50:                             return False
        # R7: identity coherence
        if identity_value < 0.40:                                     return False
        # R8: reproductive maturity
        if self.r_maturity < 0.65:                                return False
        return True

    def initiate(self):
        self._replicating = True
        self.d_stage = 0

    def develop_tick(self, met: Metabolism, genome: "Genome") -> Optional[str]:
        """
        Advances offspring development.
        Returns: 'continue' | 'success' | 'fail' | 'abort'
        """
        if not self._replicating:
            return None

        # Development cost per tick
        atp_cost = genome.maturation_cost_rate * met.a_free_cap * 0.03
        struct_cost = 1.0
        if met.a_free < atp_cost or met.m_struct < struct_cost:
            # abort if no resources
            self._abort_count += 1
            if self._abort_count > 5:
                self._replicating = False
                self._abort_count = 0
                # abort cost to parent
                met.consume_atp(atp_cost * 3)
                return 'abort'
            return 'continue'

        met.consume_atp(atp_cost)
        met.consume_structural(struct_cost)
        self.r_material = max(0, self.r_material - 0.8)
        self.d_stage += 1

        if self.d_stage >= self.d_max_stage:
            self._replicating = False
            self._abort_count = 0
            # failure from accumulated risk
            if self._rng.random() < self.r_failure_risk:
                return 'fail'
            return 'success'

        return 'continue'

    def build_offspring_genome(self, parent_genome: "Genome",
                                parent_neural: NeuralCore,
                                parent_memory: MaterialMemory,
                                rng: random.Random) -> Optional["Genome"]:
        """
        Builds offspring genome with H1-H4.
        May return None if heritable material is too degraded.
        """
        if parent_memory.h_integrity < 0.3:
            # inheritance so corrupt that offspring is nonviable
            return None

        # Base: mutation of parent genome
        child_genome = parent_genome.mutate(rng)

        # H1: structural inheritance — biased by B5's h_heritable
        h_her = parent_memory.get_heritable_snapshot()
        child_genome.membrane_strength     = float(np.clip(
            child_genome.membrane_strength * (0.7 + 0.3 * h_her[0]), 0.1, 1.0))
        child_genome.metabolic_base_rate   = float(np.clip(
            child_genome.metabolic_base_rate * (0.7 + 0.3 * h_her[1]), 0.1, 1.0))
        child_genome.conversion_efficiency = float(np.clip(
            child_genome.conversion_efficiency * (0.7 + 0.3 * h_her[2]), 0.1, 0.85))
        child_genome.repair_capacity_base  = float(np.clip(
            child_genome.repair_capacity_base * (0.7 + 0.3 * h_her[3]), 0.1, 1.0))

        # H2: regulatory inheritance — w_reg partially from parent
        w_reg_parent = parent_genome.w_reg().flatten().tolist()
        w_reg_child  = child_genome.w_reg_flat
        child_genome.w_reg_flat = [
            0.60 * p + 0.40 * c
            for p, c in zip(w_reg_parent, w_reg_child)
        ]

        # H3: cognitive inheritance — neural predispositions (not full learning)
        w_ih_pred, w_hh_pred, w_ho_pred = parent_neural.get_heritable_predispositions()
        child_genome.neural_w_ih_flat = [
            0.5 * p + 0.5 * c
            for p, c in zip(w_ih_pred, child_genome.neural_w_ih_flat)
        ]
        child_genome.neural_w_hh_flat = [
            0.5 * p + 0.5 * c
            for p, c in zip(w_hh_pred, child_genome.neural_w_hh_flat)
        ]
        child_genome.neural_w_ho_flat = [
            0.5 * p + 0.5 * c
            for p, c in zip(w_ho_pred, child_genome.neural_w_ho_flat)
        ]

        # H4: developmental inheritance — from the already-mutated genome
        # (maturation timings, costs, thresholds — already in child_genome)

        # Heritable integrity: if low → weakened offspring
        integrity_factor = parent_memory.h_integrity
        child_genome.membrane_strength     *= integrity_factor
        child_genome.repair_capacity_base  *= integrity_factor

        # Morphogenetics: inheritance 60% parent + 40% mutated (same as H2 regulatory)
        child_genome.morphogen_response_flat = [
            0.60 * p + 0.40 * c
            for p, c in zip(
                parent_genome.morphogen_response_flat,
                child_genome.morphogen_response_flat
            )
        ]

        return child_genome

    def to_dict(self) -> Dict:
        return {
            "maturity":     round(self.r_maturity, 4),
            "material":     round(self.r_material, 4),
            "replicating":  self._replicating,
            "d_stage":      self.d_stage,
            "failure_risk": round(self.r_failure_risk, 4)
        }

class Identity:
    """
    B9: Identity / individuation.
    Not a physical module. It is a systemic condition computed from all blocks.

    I = f(boundary_continuity, memory_continuity, regulatory_coherence,
           structural_coherence, causal_closure_proxy)

    If I < θI_dead → organizational death M3.

    Id = {i_boundary_continuity, i_memory_continuity, i_regulatory_coherence,
          i_structural_coherence, i_causal_closure_proxy}
    """

    THETA_I_DEAD = 0.18   # M3 threshold
    THETA_I_REP  = 0.40   # threshold for reproduction (R7)

    def __init__(self):
        self.i_boundary_continuity   = 1.0
        self.i_memory_continuity     = 1.0
        self.i_regulatory_coherence  = 1.0
        self.i_structural_coherence  = 1.0
        self.i_causal_closure_proxy  = 1.0
        self.I = 1.0
        self._history: List[float] = []

    def compute(self, boundary: Boundary, met: Metabolism,
                memory: MaterialMemory, homeostasis: HomeostasisRegulator,
                neural: NeuralCore, damage: float) -> float:
        """
        Computes I as a weighted function of individuation components.
        """
        # Boundary continuity
        self.i_boundary_continuity = boundary.c_integrity * (1 - boundary.c_permanent_damage)

        # Memory continuity
        self.i_memory_continuity = memory.h_integrity

        # Regulatory coherence
        self.i_regulatory_coherence = homeostasis.g_regulatory_coherence

        # Structural coherence (functional metabolism)
        self.i_structural_coherence = met.eta_metabolic * (1 - min(1.0, damage))

        # Organizational closure proxy:
        # Is the circuit closed? Measures whether all flows are active.
        #   M→A: metabolism produces ATP
        f_met_to_atp  = min(1.0, met.atp_produced_last_tick / 3.0)
        #   A→Rep: repair active (uses ATP)
        f_atp_to_rep  = 1.0 if met.a_free > 5.0 and damage < 0.8 else 0.3
        #   Rep→B1: boundary being maintained
        f_rep_to_bnd  = min(1.0, boundary.c_integrity / 0.5) if boundary.c_integrity > 0 else 0.0
        #   B1→M: boundary enables exchange
        f_bnd_to_met  = boundary.c_permeability_resource * boundary.c_integrity
        #   Cog→action: neural active
        f_cog_to_act  = min(1.0, neural.t_excitation / 0.1) if neural.t_excitation > 0.01 else 0.2

        # Proxy = geometric mean of the 5 flows
        flows = [f_met_to_atp, f_atp_to_rep, f_rep_to_bnd, f_bnd_to_met, f_cog_to_act]
        product = 1.0
        for f in flows:
            product *= max(0.01, f)
        self.i_causal_closure_proxy = product ** (1.0/5)

        # ── Composite I (weighted)
        self.I = (
            0.25 * self.i_boundary_continuity +
            0.20 * self.i_memory_continuity   +
            0.20 * self.i_regulatory_coherence +
            0.15 * self.i_structural_coherence +
            0.20 * self.i_causal_closure_proxy
        )
        self.I = max(0.0, min(1.0, self.I))

        self._history.append(self.I)
        if len(self._history) > 30:
            self._history.pop(0)

        return self.I

    @property
    def is_organizationally_dead(self) -> bool:
        """M3: organizational death — identity irreversibly collapsed."""
        if len(self._history) < 10:
            return False
        # only M3 if sustained below threshold
        return all(v < self.THETA_I_DEAD for v in self._history[-8:])

    def to_dict(self) -> Dict:
        return {
            "I": round(self.I, 4),
            "boundary_continuity":  round(self.i_boundary_continuity, 4),
            "memory_continuity":    round(self.i_memory_continuity, 4),
            "regulatory_coherence": round(self.i_regulatory_coherence, 4),
            "structural_coherence": round(self.i_structural_coherence, 4),
            "causal_closure_proxy": round(self.i_causal_closure_proxy, 4)
        }



# ─────────────────────────────────────────────────────────────
# B8b: BIOSEMIOTIC INTERCELLULAR COMMUNICATION
# ─────────────────────────────────────────────────────────────

class CommunicationSystem:
    """
    Intercellular communication for digital life.

    Not NLP, not embeddings, not human messaging. It is a minimal
    biosemiotic system:

    1. Cells emit costly signals to a shared medium.
    2. The world diffuses and degrades them.
    3. Other cells receive them according to boundary + heritable receptors.
    4. Reception modulates homeostasis, cognition, movement and memory.
    5. The signal only "means" something because it alters survival and behavior.

    Communicative state:
      z_received    = locally perceived signals
      z_decoded     = body translation by heritable receptor
      z_social_move = vectorial movement bias from signal gradients
      z_coherence   = recent stability of received signal
    """

    def __init__(self, genome: "Genome"):
        self.receptor_w = genome.signal_receptor_w().copy()
        self.sensitivity = genome.signal_receptor_sensitivity
        self.emission_strength = genome.signal_emission_strength
        self.selectivity = genome.signal_selectivity
        self.cost_factor = genome.signal_cost_factor

        self.z_received = np.zeros(N_SIGNAL_CHANNELS, dtype=np.float64)
        self.z_decoded = np.zeros(4, dtype=np.float64)
        self.z_social_move = np.zeros(2, dtype=np.float64)
        self.z_priority_bias = np.zeros(4, dtype=np.float64)
        self.z_coherence = 1.0
        self.z_signal_load = 0.0
        self.z_emitted_last = np.zeros(N_SIGNAL_CHANNELS, dtype=np.float64)
        self.z_received_history: List[np.ndarray] = []

    def receive(self, world: SpatialWorld, x: int, y: int,
                boundary: Boundary, met: Metabolism,
                memory: MaterialMemory) -> Dict:
        """
        Reads local signals + gradients and translates them to body effects.

        The boundary regulates signal input; damaged memory and internal noise degrade
        comprehension. tanh saturation prevents an infinite signal from controlling everything.
        """
        raw = world.sample_signals(x, y)
        permeability = boundary.c_permeability_signal * boundary.c_integrity
        memory_gate = 0.4 + 0.6 * memory.h_integrity
        metabolic_gate = 0.35 + 0.65 * met.atp_fraction
        received = raw * permeability * self.sensitivity * memory_gate * metabolic_gate

        # Biophysical normalization: strong signals saturate, they don't scale infinitely.
        norm = np.tanh(received / 18.0)
        self.z_received = norm
        self.z_signal_load = float(np.mean(norm))

        # Heritable translation to four body axes:
        # [maintenance, repair, action, reproduction].
        decoded = np.tanh(self.receptor_w @ norm)
        self.z_decoded = decoded

        # Gradients: the cell doesn't just "hear" intensity; it detects direction.
        def grad(ch):
            dx, dy = world.signal_gradient(x, y, ch)
            return np.array([dx, dy], dtype=np.float64)

        g_resource = grad("nutrient_beacon")
        g_danger   = grad("toxin_alarm") + 0.8 * grad("death_trace")
        g_crowd    = grad("crowding")
        g_repro    = grad("reproduction_ready")

        social = (
            0.90 * g_resource -
            1.15 * g_danger -
            0.65 * g_crowd +
            0.15 * g_repro
        )
        norm_social = np.linalg.norm(social)
        if norm_social > 1e-9:
            social = social / norm_social
        self.z_social_move = np.clip(social * self.sensitivity, -1.0, 1.0)

        # Priority bias: the social does not replace homeostasis; it perturbs it.
        # Order: [maintenance, repair, action, reproduction].
        self.z_priority_bias = np.array([
            0.08 * norm[SIGNAL_IDX["energy_need"]] + 0.04 * norm[SIGNAL_IDX["death_trace"]],
            0.10 * norm[SIGNAL_IDX["repair_need"]] + 0.08 * norm[SIGNAL_IDX["toxin_alarm"]],
            0.07 * norm[SIGNAL_IDX["nutrient_beacon"]] - 0.05 * norm[SIGNAL_IDX["crowding"]],
            0.06 * norm[SIGNAL_IDX["reproduction_ready"]] - 0.04 * norm[SIGNAL_IDX["death_trace"]],
        ], dtype=np.float64)
        self.z_priority_bias += decoded * 0.03
        self.z_priority_bias = np.clip(self.z_priority_bias, -0.15, 0.20)

        # Coherence = stability of what is received; if it changes violently,
        # the cell cannot "understand" with confidence.
        self.z_received_history.append(norm.copy())
        if len(self.z_received_history) > 16:
            self.z_received_history.pop(0)
        if len(self.z_received_history) >= 4:
            arr = np.array(self.z_received_history[-4:])
            self.z_coherence = float(max(0.0, 1.0 - np.mean(np.std(arr, axis=0)) * 3.5))

        return {
            "received": self.z_received,
            "decoded": self.z_decoded,
            "social_move": self.z_social_move,
            "priority_bias": self.z_priority_bias,
            "coherence": self.z_coherence,
            "signal_load": self.z_signal_load,
        }

    def modulate_homeostasis(self, homeostasis: HomeostasisRegulator):
        """
        Integrates social signals into B3 without overwriting the regulatory network.
        It is a small, normalized perturbation, not a dominant external if.
        """
        p = np.array([
            homeostasis.p_maintenance,
            homeostasis.p_repair,
            homeostasis.p_action,
            homeostasis.p_reproduction,
        ], dtype=np.float64)
        p = np.clip(p + self.z_priority_bias, 0.001, 1.0)
        p = p / (p.sum() + 1e-9)
        homeostasis.p_maintenance  = float(p[0])
        homeostasis.p_repair       = float(p[1])
        homeostasis.p_action       = float(p[2])
        homeostasis.p_reproduction = float(p[3])

    def neural_environment_bias(self, nut_local: float, tox_local: float,
                                damage: float) -> Tuple[float, float, float]:
        """
        Returns biases for the three NeuralCore environmental inputs:
        nutrient, toxin, damage. Maintains dimensionality 6 to not break B4.
        """
        resource = min(1.0, nut_local / 80.0)
        toxin = min(1.0, tox_local / 40.0)
        dmg = min(1.0, damage)

        resource = np.clip(resource + 0.22 * self.z_received[SIGNAL_IDX["nutrient_beacon"]], 0, 1)
        toxin = np.clip(
            toxin +
            0.25 * self.z_received[SIGNAL_IDX["toxin_alarm"]] +
            0.18 * self.z_received[SIGNAL_IDX["death_trace"]],
            0, 1
        )
        dmg = np.clip(
            dmg +
            0.14 * self.z_received[SIGNAL_IDX["repair_need"]] +
            0.10 * self.z_received[SIGNAL_IDX["energy_need"]],
            0, 1
        )
        return float(resource), float(toxin), float(dmg)

    def emit(self, world: SpatialWorld, x: int, y: int, met: Metabolism,
             boundary: Boundary, homeostasis: HomeostasisRegulator,
             memory: MaterialMemory, reproduction: ReproductionModule,
             identity: Identity, damage: float, nut_local: float, tox_local: float,
             phase: "LifePhase") -> Dict:
        """
        Emits signals anchored to vital state. Emitting costs ATP, so it is not
        free telepathy or artificial broadcast.
        """
        v = np.zeros(N_SIGNAL_CHANNELS, dtype=np.float64)

        atp = met.atp_fraction
        waste = met.waste_fraction

        # Positive signals: only if the cell is stable, to avoid accidental
        # lies from collapsed organisms.
        stability = boundary.c_integrity * memory.h_integrity * identity.I
        v[SIGNAL_IDX["nutrient_beacon"]] = max(0.0, min(1.0, nut_local / 80.0)) * stability

        # Alarms and needs: come from real homeostatic errors.
        v[SIGNAL_IDX["toxin_alarm"]] = np.clip(tox_local / 40.0 + waste * 0.7 + damage * 0.5, 0, 1)
        v[SIGNAL_IDX["energy_need"]] = np.clip(homeostasis.g_energy_error + max(0, 0.28 - atp), 0, 1)
        v[SIGNAL_IDX["repair_need"]] = np.clip(homeostasis.g_damage_error + homeostasis.g_boundary_error +
                                               (1.0 - memory.h_integrity) * 0.35, 0, 1)

        v[SIGNAL_IDX["reproduction_ready"]] = np.clip(
            reproduction.r_maturity * (1.0 - damage) * identity.I *
            (1.0 if not reproduction._replicating else 0.25),
            0, 1
        )

        # Living cell occupies space: communicates local ecological saturation.
        v[SIGNAL_IDX["crowding"]] = 0.15 + 0.35 * self.z_received[SIGNAL_IDX["crowding"]]

        # Dying cells emit collapse trace before death.
        dying = 1.0 if phase in (LifePhase.DYING, LifePhase.STRESSED) else 0.0
        v[SIGNAL_IDX["death_trace"]] = np.clip(dying * (1.0 - identity.I + damage), 0, 1)

        # Selectivity: reduces noise from weak signals; lineages can be more
        # "talkative" or more "reserved".
        threshold = 0.08 + (1.0 - min(1.0, self.selectivity)) * 0.10
        v[v < threshold] = 0.0

        # Energy cost proportional to emitted load.
        load = float(np.sum(v))
        cost = load * self.cost_factor * (1.0 + 0.5 * homeostasis.g_stress)
        paid = met.consume_atp(cost)
        if cost > 1e-9:
            v *= min(1.0, paid / cost)

        # Local deposit. Multipliers convert [0,1] to concentration.
        for i, amount in enumerate(v):
            if amount > 0:
                world.deposit_signal(x, y, i, amount * self.emission_strength * 8.0)

        self.z_emitted_last = v
        return {"emitted": v, "cost": cost, "paid": paid}

    def repair(self, repair_amount: float):
        """B6 can also stabilize communicative receptors."""
        self.z_coherence = min(1.0, self.z_coherence + repair_amount * 0.04)
        self.receptor_w = np.clip(self.receptor_w, -3, 3)

    def to_dict(self) -> Dict:
        return {
            "channels": list(SIGNAL_CHANNELS),
            "received": {name: round(float(self.z_received[i]), 3)
                         for i, name in enumerate(SIGNAL_CHANNELS)},
            "emitted_last": {name: round(float(self.z_emitted_last[i]), 3)
                             for i, name in enumerate(SIGNAL_CHANNELS)},
            "decoded": [round(float(v), 3) for v in self.z_decoded],
            "social_move": [round(float(v), 3) for v in self.z_social_move],
            "priority_bias": [round(float(v), 3) for v in self.z_priority_bias],
            "coherence": round(float(self.z_coherence), 3),
            "signal_load": round(float(self.z_signal_load), 3),
        }


# ─────────────────────────────────────────────────────────────
# LIFE PHASE
# ─────────────────────────────────────────────────────────────

class LifePhase(Enum):
    DEVELOPING  = "developing"
    ACTIVE      = "active"
    STRESSED    = "stressed"
    REPAIRING   = "repairing"
    REPLICATING = "replicating"
    AGING       = "aging"
    DYING       = "dying"
    DEAD        = "dead"


class CellType(Enum):
    STEM      = "stem"
    BOUNDARY  = "boundary"
    METABOLIC = "metabolic"
    REPAIR    = "repair"
    SIGNALING = "signaling"
    NEURON    = "neuron"
    SENSORY   = "sensory"
    MOTOR     = "motor"
    GERMLINE  = "germline"
    POLICING  = "policing"


class JunctionKind(Enum):
    ADHESION  = "adhesion"
    GAP       = "gap"
    METABOLIC = "metabolic"
    SYNAPTIC  = "synaptic"


@dataclass
class Junction:
    """
    Persistent relationship between two cells.

    This is the first object that converts field signals into biological
    topology: a bond with cost, stability, capacity, and minimal memory.
    """
    id: str
    cell_a: str
    cell_b: str
    kind: JunctionKind
    strength: float = 0.25
    age: int = 0
    stability: float = 0.5
    transport_capacity: float = 0.0
    signal_conductance: float = 0.0
    maintenance_cost: float = 0.02
    last_activity: int = 0
    weight: float = 0.0
    delay: int = 1
    neurotransmitter: str = "generic"
    receptor_type: str = "excitatory"
    last_pre_tick: Optional[int] = None
    last_post_tick: Optional[int] = None
    prune_score: float = 0.0
    utility_trace: float = 0.0
    event_queue: List[Tuple[int, float]] = field(default_factory=list)

    def other(self, cell_id: str) -> Optional[str]:
        if cell_id == self.cell_a:
            return self.cell_b
        if cell_id == self.cell_b:
            return self.cell_a
        return None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "a": self.cell_a,
            "b": self.cell_b,
            "kind": self.kind.value,
            "strength": round(self.strength, 3),
            "age": self.age,
            "stability": round(self.stability, 3),
            "transport_capacity": round(self.transport_capacity, 3),
            "signal_conductance": round(self.signal_conductance, 3),
            "weight": round(self.weight, 3),
            "utility": round(self.utility_trace, 3),
            "queue": len(self.event_queue),
        }


@dataclass
class OrganismInstance:
    """Causal multicellular unit derived from a connected component."""
    organism_id: str
    member_cell_ids: Set[str]
    junction_ids: Set[str]
    shared_stress: float
    boundary_integrity: float
    boundary_closure: float
    collective_energy_pressure: float
    collective_damage: float
    collective_identity: float
    development_stage: str
    reproduction_pressure: float
    role_coverage: float
    topology_integrity: float
    metabolic_exchange: float
    neural_coordination: float
    collective_motor_output: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))

    def evolutionary_score(self) -> float:
        size_term = math.log1p(len(self.member_cell_ids))
        return float(max(0.0, size_term * self.collective_identity *
                         (0.45 + self.role_coverage) *
                         (0.50 + self.topology_integrity) *
                         (1.0 - 0.55 * self.collective_damage)))

    def to_dict(self) -> Dict:
        return {
            "id": self.organism_id,
            "members": len(self.member_cell_ids),
            "junctions": len(self.junction_ids),
            "shared_stress": round(self.shared_stress, 3),
            "boundary_integrity": round(self.boundary_integrity, 3),
            "boundary_closure": round(self.boundary_closure, 3),
            "energy_pressure": round(self.collective_energy_pressure, 3),
            "collective_damage": round(self.collective_damage, 3),
            "collective_identity": round(self.collective_identity, 3),
            "development_stage": self.development_stage,
            "reproduction_pressure": round(self.reproduction_pressure, 3),
            "role_coverage": round(self.role_coverage, 3),
            "topology_integrity": round(self.topology_integrity, 3),
            "metabolic_exchange": round(self.metabolic_exchange, 3),
            "neural_coordination": round(self.neural_coordination, 3),
            "evolutionary_score": round(self.evolutionary_score(), 3),
            "collective_motor_output": [round(v, 3) for v in self.collective_motor_output],
        }


@dataclass
class OrganismState:
    """
    Causal state of the multicellular organization.

    Does not replace cells. Measures whether a superior unit exists with
    its own topology, exchange, roles, and regulatory pressure.
    """
    organism_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    member_cell_ids: Set[str] = field(default_factory=set)
    junction_ids: Set[str] = field(default_factory=set)
    shared_stress: float = 0.0
    boundary_integrity: float = 0.0
    collective_energy_pressure: float = 0.0
    collective_damage: float = 0.0
    collective_identity: float = 0.0
    development_stage: str = "solitary"
    reproduction_pressure: float = 0.0
    role_coverage: float = 0.0
    topology_integrity: float = 0.0
    boundary_closure: float = 0.0
    metabolic_exchange: float = 0.0
    neural_coordination: float = 0.0
    organisms: List[OrganismInstance] = field(default_factory=list)
    _previous_members: Dict[str, Set[str]] = field(default_factory=dict)

    BODY_JUNCTIONS = (JunctionKind.ADHESION, JunctionKind.GAP, JunctionKind.METABOLIC)

    def _reset_extinct(self):
        self.member_cell_ids = set()
        self.junction_ids = set()
        self.shared_stress = 0.0
        self.boundary_integrity = 0.0
        self.boundary_closure = 0.0
        self.collective_energy_pressure = 1.0
        self.collective_damage = 1.0
        self.collective_identity = 0.0
        self.development_stage = "extinct"
        self.reproduction_pressure = 0.0
        self.role_coverage = 0.0
        self.topology_integrity = 0.0
        self.metabolic_exchange = 0.0
        self.neural_coordination = 0.0
        self.organisms = []

    def _connected_components(self, alive_ids: Set[str],
                              junctions: Dict[str, Junction]) -> List[Set[str]]:
        adjacency = self._body_adjacency(alive_ids, junctions)
        seen: Set[str] = set()
        components: List[Set[str]] = []
        for cid in sorted(alive_ids):
            if cid not in seen:
                components.append(self._walk_component(cid, adjacency, seen))
        return components

    def _body_adjacency(self, alive_ids: Set[str],
                        junctions: Dict[str, Junction]) -> Dict[str, Set[str]]:
        adjacency: Dict[str, Set[str]] = {cid: set() for cid in alive_ids}
        for j in junctions.values():
            if j.kind in self.BODY_JUNCTIONS and j.cell_a in alive_ids and j.cell_b in alive_ids:
                adjacency[j.cell_a].add(j.cell_b)
                adjacency[j.cell_b].add(j.cell_a)
        return adjacency

    def _walk_component(self, start: str, adjacency: Dict[str, Set[str]],
                        seen: Set[str]) -> Set[str]:
        stack = [start]
        component: Set[str] = set()
        seen.add(start)
        while stack:
            cur = stack.pop()
            component.add(cur)
            unseen_neighbors = adjacency[cur] - seen
            seen.update(unseen_neighbors)
            stack.extend(unseen_neighbors)
        return component

    def _stable_id_for(self, comp: Set[str], used: Set[str]) -> str:
        best_id = None
        best_overlap = 0
        for oid, old_members in self._previous_members.items():
            if oid in used:
                continue
            overlap = len(comp & old_members)
            if overlap > best_overlap:
                best_id = oid
                best_overlap = overlap
        if best_id is not None and best_overlap / max(1, len(comp)) >= 0.25:
            used.add(best_id)
            return best_id
        oid = str(uuid.uuid4())[:8]
        used.add(oid)
        return oid

    def _compute_instance(self, oid: str, comp: Set[str], cells: Dict[str, "Cell"],
                          junctions: Dict[str, Junction]) -> OrganismInstance:
        members = [cells[cid] for cid in comp if cid in cells and cells[cid].alive]
        n = len(members)
        comp_junction_ids, active_junctions, body_junctions, synapses = self._instance_junctions(comp, junctions)
        shared_stress = float(np.mean([c.homeostasis.g_stress for c in members]))
        energy_pressure = float(np.mean([1.0 - c.metabolism.atp_fraction for c in members]))
        collective_damage = float(np.mean([c.damage_x for c in members]))
        boundary_integrity, boundary_closure = self._boundary_metrics(members)
        possible_edges = max(1, n - 1)
        topology_integrity = float(np.clip(
            len(body_junctions) / possible_edges *
            (np.mean([j.strength for j in body_junctions]) if body_junctions else 0.0),
            0.0, 1.0
        ))
        present_roles, role_coverage = self._role_metrics(members)
        metabolic_exchange, neural_coordination = self._exchange_and_coordination(
            n, active_junctions, synapses, present_roles
        )
        mean_identity = float(np.mean([c.identity.I for c in members]))
        collective_identity = self._collective_identity(
            mean_identity, topology_integrity, metabolic_exchange, role_coverage,
            boundary_closure, neural_coordination, shared_stress
        )
        germline_ready = any(c.cell_type == CellType.GERMLINE and
                             c.reproduction.r_maturity > 0.7 and c.identity.I > 0.55
                             for c in members)
        reproduction_pressure = float(np.clip(
            (1.0 if germline_ready else 0.0) *
            collective_identity *
            (1.0 - energy_pressure) *
            (0.55 + 0.45 * role_coverage),
            0.0, 1.0
        ))
        stage = self._organism_stage(
            n, body_junctions, collective_identity, role_coverage,
            boundary_closure, neural_coordination
        )
        cmo = self._collective_motor_output(members)

        return OrganismInstance(
            organism_id=oid,
            member_cell_ids=set(comp),
            junction_ids=comp_junction_ids,
            shared_stress=shared_stress,
            boundary_integrity=boundary_integrity,
            boundary_closure=boundary_closure,
            collective_energy_pressure=energy_pressure,
            collective_damage=collective_damage,
            collective_identity=collective_identity,
            development_stage=stage,
            reproduction_pressure=reproduction_pressure,
            role_coverage=role_coverage,
            topology_integrity=topology_integrity,
            metabolic_exchange=metabolic_exchange,
            neural_coordination=neural_coordination,
            collective_motor_output=cmo,
        )

    def _instance_junctions(self, comp: Set[str], junctions: Dict[str, Junction]):
        comp_junction_ids = {
            jid for jid, j in junctions.items()
            if j.cell_a in comp and j.cell_b in comp
        }
        active_junctions = [junctions[jid] for jid in comp_junction_ids]
        body_junctions = [j for j in active_junctions if j.kind in self.BODY_JUNCTIONS]
        synapses = [j for j in active_junctions if j.kind == JunctionKind.SYNAPTIC]
        return comp_junction_ids, active_junctions, body_junctions, synapses

    def _boundary_metrics(self, members: List["Cell"]) -> Tuple[float, float]:
        boundary_cells = [c for c in members if c.cell_type == CellType.BOUNDARY]
        boundary_integrity = float(np.mean([c.boundary.c_integrity for c in boundary_cells])
                                   if boundary_cells else 0.0)
        xs = [c.x for c in members]
        ys = [c.y for c in members]
        bbox_area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1) if members else 1
        expected_surface = max(1.0, 2.0 * math.sqrt(max(1, bbox_area)))
        boundary_closure = float(np.clip(
            (len(boundary_cells) / expected_surface) * boundary_integrity,
            0.0, 1.0
        ))
        return boundary_integrity, boundary_closure

    def _role_metrics(self, members: List["Cell"]) -> Tuple[Set[CellType], float]:
        required_roles = {
            CellType.BOUNDARY, CellType.METABOLIC, CellType.REPAIR,
            CellType.SIGNALING, CellType.NEURON
        }
        present_roles = {c.cell_type for c in members}
        return present_roles, len(required_roles & present_roles) / len(required_roles)

    def _exchange_and_coordination(self, n: int, active_junctions: List[Junction],
                                   synapses: List[Junction], present_roles: Set[CellType]) -> Tuple[float, float]:
        metabolic_exchange = float(np.clip(
            sum(j.transport_capacity * j.strength for j in active_junctions
                if j.kind in (JunctionKind.GAP, JunctionKind.METABOLIC)) / max(1.0, n),
            0.0, 1.0
        ))
        neural_roles = {CellType.SENSORY, CellType.NEURON, CellType.MOTOR}
        role_loop = len(neural_roles & present_roles) / len(neural_roles)
        neural_coordination = float(np.clip(
            (len(synapses) / max(1.0, n)) *
            (np.mean([abs(j.weight) * j.signal_conductance * j.strength
                      for j in synapses]) if synapses else 0.0) *
            (0.5 + role_loop),
            0.0, 1.0
        ))
        return metabolic_exchange, neural_coordination

    def _collective_identity(self, mean_identity: float, topology_integrity: float,
                             metabolic_exchange: float, role_coverage: float,
                             boundary_closure: float, neural_coordination: float,
                             shared_stress: float) -> float:
        return float(np.clip(
            0.22 * mean_identity +
            0.18 * topology_integrity +
            0.16 * metabolic_exchange +
            0.16 * role_coverage +
            0.14 * boundary_closure +
            0.08 * neural_coordination +
            0.06 * (1.0 - shared_stress),
            0.0, 1.0
        ))

    def _organism_stage(self, n: int, body_junctions: List[Junction],
                        collective_identity: float, role_coverage: float,
                        boundary_closure: float, neural_coordination: float) -> str:
        if n < 3 or not body_junctions:
            return "solitary"
        if collective_identity < 0.35:
            return "aggregate"
        if role_coverage < 0.6 or boundary_closure < 0.20:
            return "proto_tissue"
        if neural_coordination > 0.18 and boundary_closure > 0.35:
            return "integrated_body"
        return "integrated"

    def _collective_motor_output(self, members: List["Cell"]) -> Tuple[float, float, float]:
        neural_motor = [c for c in members
                        if c.cell_type in (CellType.NEURON, CellType.MOTOR)]
        if not neural_motor:
            return (0.0, 0.0, 0.0)
        weight_total = sum(c.identity.I for c in neural_motor) + 1e-9
        motor = [
            sum(float(c.neural.t_action_bias[i]) * c.identity.I for c in neural_motor) / weight_total
            for i in range(3)
        ]
        return tuple(float(np.clip(v, -1.0, 1.0)) for v in motor)

    def update(self, cells: Dict[str, "Cell"], junctions: Dict[str, Junction]):
        alive = [c for c in cells.values() if c.alive]
        if not alive:
            self._reset_extinct()
            return

        alive_ids = {c.id for c in alive}
        components = self._connected_components(alive_ids, junctions)
        used_ids: Set[str] = set()
        organisms = [
            self._compute_instance(self._stable_id_for(comp, used_ids), comp, cells, junctions)
            for comp in components
        ]
        organisms.sort(key=lambda o: (o.evolutionary_score(), len(o.member_cell_ids)), reverse=True)
        self.organisms = organisms
        self._previous_members = {o.organism_id: set(o.member_cell_ids) for o in organisms}

        for c in alive:
            c.organism_id = None
        for org in organisms:
            for cid in org.member_cell_ids:
                if cid in cells:
                    cells[cid].organism_id = org.organism_id

        primary = organisms[0]
        self.organism_id = primary.organism_id
        self.member_cell_ids = set(primary.member_cell_ids)
        self.junction_ids = set(primary.junction_ids)
        self.shared_stress = primary.shared_stress
        self.boundary_integrity = primary.boundary_integrity
        self.boundary_closure = primary.boundary_closure
        self.collective_energy_pressure = primary.collective_energy_pressure
        self.collective_damage = primary.collective_damage
        self.collective_identity = primary.collective_identity
        self.development_stage = primary.development_stage
        self.reproduction_pressure = primary.reproduction_pressure
        self.role_coverage = primary.role_coverage
        self.topology_integrity = primary.topology_integrity
        self.metabolic_exchange = primary.metabolic_exchange
        self.neural_coordination = primary.neural_coordination

    def get(self, organism_id: Optional[str]) -> Optional[OrganismInstance]:
        if organism_id is None:
            return None
        for org in self.organisms:
            if org.organism_id == organism_id:
                return org
        return None

    def to_dict(self) -> Dict:
        return {
            "id": self.organism_id,
            "members": len(self.member_cell_ids),
            "organism_count": len(self.organisms),
            "junctions": len(self.junction_ids),
            "shared_stress": round(self.shared_stress, 3),
            "boundary_integrity": round(self.boundary_integrity, 3),
            "boundary_closure": round(self.boundary_closure, 3),
            "energy_pressure": round(self.collective_energy_pressure, 3),
            "collective_damage": round(self.collective_damage, 3),
            "collective_identity": round(self.collective_identity, 3),
            "development_stage": self.development_stage,
            "reproduction_pressure": round(self.reproduction_pressure, 3),
            "role_coverage": round(self.role_coverage, 3),
            "topology_integrity": round(self.topology_integrity, 3),
            "metabolic_exchange": round(self.metabolic_exchange, 3),
            "neural_coordination": round(self.neural_coordination, 3),
            "organisms": [o.to_dict() for o in self.organisms[:8]],
        }


# ─────────────────────────────────────────────────────────────
# CELL — integrates the 9 blocks
# ─────────────────────────────────────────────────────────────

class Cell:
    """
    CNDV: Living Digital Neural Cell.

    Integrates B1-B9 in strict tick-order from the architectural document:
    1. exchange with environment       (B1 + B8)
    2. metabolic transformation        (B2)
    3. waste and damage accumulation   (D1-D7)
    4. homeostatic update              (B3)
    5. prioritized repair              (B6)
    6. cognition and action            (B4)
    7. memory update                   (B5)
    8. viability evaluation            (B9)
    9. reproduction evaluation         (B7)
    10. death if organization is unrecoverable (M1, M2, M3)
    """

    def __init__(self, x: int, y: int, world: SpatialWorld,
                 genome: Optional[Genome] = None,
                 rng: random.Random = None,
                 developing: bool = False):
        self.id    = str(uuid.uuid4())[:8]
        self.x     = x % world.W
        self.y     = y % world.H
        self.world = world
        self.rng   = rng or random.Random()
        self.genome = genome or Genome.create(self.rng)

        # ── 9 Blocks
        self.boundary    = Boundary(
            c_integrity=0.6 if developing else 1.0,
            c_permeability_resource=0.7 * self.genome.transport_capacity,
            c_transport_capacity=self.genome.transport_capacity,
            c_maintenance_cost=0.04 + (1 - self.genome.membrane_strength) * 0.03
        )
        self.metabolism  = Metabolism(
            r_raw=28.0 if developing else 30.0,
            a_free=34.0 if developing else 40.0,
        )
        self.homeostasis = HomeostasisRegulator(self.genome)
        self.neural      = NeuralCore(self.genome)
        self.memory      = MaterialMemory(self.genome)
        self.communication = CommunicationSystem(self.genome)
        self.repair      = RepairSystem(self.genome)
        self.reproduction = ReproductionModule(self.genome)
        self.identity    = Identity()

        # Global state S(t)
        self.damage_x    = 0.0   # total accumulated damage
        self.age_ticks   = 0
        self.phase       = LifePhase.DEVELOPING if developing else LifePhase.ACTIVE
        self.alive       = True
        self.death_cause: Optional[str] = None
        self.cell_type   = CellType.STEM if developing else CellType.METABOLIC
        self.type_commitment = 0.15 if developing else 0.35

        # Multicellular layer: persistent bonds, cooperation and surveillance.
        self.junction_ids: Set[str] = set()
        self.attachment_strength = 0.0
        self.organism_id: Optional[str] = None
        self.synaptic_input = np.zeros(4, dtype=np.float64)
        self.last_spike_tick: Optional[int] = None
        self.spike_output = 0.0
        self._refractory_ticks = 0
        self.contribution_score = 0.0
        self.received_support = 0.0
        self.provided_support = 0.0
        self.cheater_score = 0.0
        self.unauthorized_reproduction_attempts = 0

        # History
        self.event_log: List[str] = []
        self.metrics_history: List[Dict] = []
        self._starving_ticks = 0
        self._generation = 0

        # Feature G: epigenetic cell fate bias — set by parent during division.
        # Biases differentiation toward the parent's specialized type, enabling
        # clonal expansion of neural and other specialized tissue.
        self._parent_cell_type: Optional[CellType] = None

        world.register(x, y, self.id)

    # ─────────────────────────────────────────────────────────
    # MAIN TICK
    # ─────────────────────────────────────────────────────────

    def tick(self) -> Optional["Cell"]:
        """
        Executes a full tick in the order of the architectural document.
        Returns an offspring cell if reproduction is successful, None otherwise.
        """
        if not self.alive:
            return None

        self.age_ticks += 1
        offspring = None

        # ──────────────────────────────────────────────────────
        # STEP 1: Exchange with environment (B1 + B8)
        # ──────────────────────────────────────────────────────
        nut_local, tox_local = self.world.sample(self.x, self.y)

        # Situated communicative reception: the cell perceives the social field before
        # deciding on capture, repair, and movement.
        self.communication.receive(
            self.world, self.x, self.y,
            self.boundary, self.metabolism, self.memory
        )

        # The boundary filters what enters
        # Incoming resource: modulated by integrity + transport capacity + neural_gate
        capture_base = (nut_local * self.boundary.c_permeability_resource *
                        self.boundary.c_transport_capacity)

        # The neural module will contribute its capture_modulation after step 6,
        # but we use the last available signal (from the previous tick)
        capture_mod = float((self.neural.t_action_bias[2] + 1) / 2) if np.any(
            self.neural.t_action_bias != 0) else 0.5

        # Actual capture — without double transport multiplication
        social_capture = 1.0 + 0.18 * self.communication.z_received[SIGNAL_IDX["nutrient_beacon"]]
        resource_captured = min(nut_local * 0.60, capture_base * (0.6 + 0.4 * capture_mod) * social_capture)
        resource_captured = self.world.consume_nutrient(self.x, self.y, resource_captured)
        self.metabolism.r_raw = min(self.metabolism.r_raw_cap,
                                    self.metabolism.r_raw + resource_captured)

        # Toxins entering through damaged boundary
        tox_entering = tox_local * self.boundary.c_permeability_toxin
        self.metabolism.w_waste = min(
            self.metabolism.w_waste_cap,
            self.metabolism.w_waste + tox_entering * 0.3
        )

        # ──────────────────────────────────────────────────────
        # STEP 2: Metabolic transformation (B2)
        # ──────────────────────────────────────────────────────
        self.metabolism.step(
            self.genome,
            tox_internal=self.metabolism.w_waste,
            maintenance_priority=self.homeostasis.p_maintenance
        )

        # Waste excretion to the environment
        waste_excreted = self.metabolism.w_waste * 0.04 * self.boundary.c_integrity
        self.metabolism.w_waste -= waste_excreted
        self.world.deposit_toxin(self.x, self.y, waste_excreted * 0.5)

        # ──────────────────────────────────────────────────────
        # STEP 3: Damage accumulation (D1-D7)
        # ──────────────────────────────────────────────────────
        # D1: Boundary
        dmg_boundary = self.boundary.degrade(
            tox_local, self.metabolism.w_waste, self.age_ticks, self.genome
        )

        # D2: Metabolic — already applied inside metabolism.step()

        # D6: Ecological — local scarcity degrades gradually
        if nut_local < 5.0:
            self.damage_x = min(1.0, self.damage_x + 0.003)

        # D7: Organizational — if closure broken, diffuse damage is added
        if self.identity.i_causal_closure_proxy < 0.3:
            self.damage_x = min(1.0, self.damage_x + 0.005)

        # Total accumulated damage
        self.damage_x = min(1.0, self.damage_x + dmg_boundary * 0.5)

        # Boundary maintenance cost
        self.metabolism.consume_atp(
            self.boundary.c_maintenance_cost * self.metabolism.a_free_cap * 0.01
        )

        # ──────────────────────────────────────────────────────
        # STEP 4: Homeostatic update (B3)
        # ──────────────────────────────────────────────────────
        mem_reg_bias = self.memory.get_regulatory_bias()  # B5 → B3
        hom_out = self.homeostasis.update(
            self.metabolism, self.boundary,
            self.damage_x, self.memory.h_integrity,
            mem_reg_bias
        )
        # Regulatory plasticity: w_reg learns slowly
        self.homeostasis.adapt_w_reg(
            learning_signal=self.memory.h_integrity * 0.1
        )
        # Social signals gently bias homeostasis, without replacing it.
        self.communication.modulate_homeostasis(self.homeostasis)

        # ──────────────────────────────────────────────────────
        # STEP 5: Prioritized repair (B6)
        # ──────────────────────────────────────────────────────
        dmg_reduced, _ = self.repair.execute(
            self.metabolism, self.boundary, self.neural, self.memory,
            self.homeostasis, self.damage_x, self.genome
        )
        self.damage_x = max(0.0, self.damage_x - dmg_reduced)
        if dmg_reduced > 0:
            self.communication.repair(dmg_reduced)

        # ──────────────────────────────────────────────────────
        # STEP 6: Cognition and action (B4)
        # ──────────────────────────────────────────────────────
        # Normalized inputs from internal state and environment
        sig_nut, sig_tox, sig_damage = self.communication.neural_environment_bias(
            nut_local, tox_local, self.damage_x
        )
        neural_inputs = np.array([
            self.metabolism.atp_fraction,
            self.boundary.c_integrity,
            self.metabolism.waste_fraction,
            sig_nut,
            sig_tox,
            sig_damage
        ], dtype=np.float64)

        adaptive_bias = self.memory.get_adaptive_bias()  # B5 → B4
        if np.any(self.synaptic_input):
            # Direct intercellular input. Injected as internal bias, not as
            # fake nutrient/toxin, to keep physiology and synapses separate.
            adaptive_bias = adaptive_bias + np.clip(self.synaptic_input, -1.0, 1.0)
            self.synaptic_input *= 0.35

        neural_out = self.neural.step(
            neural_inputs,
            adaptive_bias,
            self.metabolism.a_free
        )
        self._update_spike_output()

        # Action: neurally guided movement + chemotaxis (B8)
        self._move(neural_out)

        # Boundary modulation by cognition (F3: B4 → B1)
        gate_signal = float((neural_out['capture_modulation'] +
                            self.boundary.c_integrity) / 2)
        self.boundary.apply_neural_gate(gate_signal)

        # Cognitive cost (real but contained)
        self.metabolism.consume_atp(
            self.neural.t_excitation * 0.3 * self.homeostasis.p_action
        )

        # ──────────────────────────────────────────────────────
        # STEP 7: Memory update (B5)
        # ──────────────────────────────────────────────────────
        mem_out = self.memory.update(
            boundary_integrity=self.boundary.c_integrity,
            homeostatic_errors=hom_out.get('errors', {}),
            neural_trace=self.neural.t_adaptive_trace,
            atp_available=self.metabolism.a_free,
            waste_internal=self.metabolism.w_waste,
            damage=self.damage_x
        )
        # Pay memory maintenance cost
        self.metabolism.consume_atp(mem_out.get('maintenance_cost', 0))

        # ──────────────────────────────────────────────────────
        # STEP 8: Viability evaluation (B9)
        # ──────────────────────────────────────────────────────
        I = self.identity.compute(
            self.boundary, self.metabolism, self.memory,
            self.homeostasis, self.neural, self.damage_x
        )

        # Update phase according to state
        self._update_phase()

        # Communicative emission: occurs after computing identity so that
        # the sent signal reflects the current organizational state.
        self.communication.emit(
            self.world, self.x, self.y,
            self.metabolism, self.boundary, self.homeostasis,
            self.memory, self.reproduction, self.identity,
            self.damage_x, nut_local, tox_local, self.phase
        )

        # ──────────────────────────────────────────────────────
        # STEP 9: Reproduction evaluation (B7)
        # ──────────────────────────────────────────────────────
        self.reproduction.tick_maturity(self.age_ticks, self.genome,
                                        self.metabolism, self.damage_x)

        offspring = self._handle_reproduction(I)

        # ──────────────────────────────────────────────────────
        # STEP 10: Death check (M1, M2, M3)
        # ──────────────────────────────────────────────────────
        self._check_death()

        # Metrics recording
        self._record_metrics()

        return offspring

    def _update_spike_output(self):
        if self._refractory_ticks > 0:
            self.spike_output = 0.0
            self._refractory_ticks -= 1
            self.neural.update_homeostatic_scale(False)
            return
        if self.neural.t_excitation <= EXCIT_THRESHOLD:
            self.spike_output = 0.0
            self.neural.update_homeostatic_scale(False)
            return
        self.spike_output = float(np.clip(
            (self.neural.t_excitation - EXCIT_THRESHOLD) * 3.0, 0.0, 1.0))
        self._refractory_ticks = REFRACTORY_PERIOD
        self.neural.update_homeostatic_scale(True)

    def _handle_reproduction(self, identity_value: float) -> Optional["Cell"]:
        if not self.reproduction._replicating and self.reproduction.can_reproduce(
            self.metabolism, self.memory, self.damage_x, identity_value, self.genome
        ):
            self.reproduction.initiate()
            self._log("REPRODUCTION_INITIATED")
        if not self.reproduction._replicating:
            return None
        return self._advance_reproduction()

    def _advance_reproduction(self) -> Optional["Cell"]:
        result = self.reproduction.develop_tick(self.metabolism, self.genome)
        if result == 'success':
            offspring = self._spawn_offspring()
            self._log(f"OFFSPRING:{offspring.id}" if offspring else "OFFSPRING_FAILED:genome_corrupted")
            return offspring
        if result == 'fail':
            self._log("REPRODUCTION_FAILED:risk_exceeded")
        elif result == 'abort':
            self._log("REPRODUCTION_ABORTED:no_resources")
        return None
    def _move(self, neural_out: Dict):
        """
        B8: Movement guided by neural output + chemotaxis.
        The cell integrates its own signals (B4) with world gradients.
        """
        if self.metabolism.a_free < 2.0:
            return  # no energy, no movement

        # Nutrient and toxin gradients
        dnut_x, dnut_y = self.world.gradient(self.x, self.y, self.world.nutrients)
        dtox_x, dtox_y = self.world.gradient(self.x, self.y, self.world.toxins)

        # Neural movement signal (B4 output)
        neural_x = neural_out['move_x'] * self.genome.motility
        neural_y = neural_out['move_y'] * self.genome.motility

        # Chemotaxis: attraction to nutrients, repulsion from toxins
        chemotaxis_x = (dnut_x * self.genome.chemotaxis_gain -
                        dtox_x * self.genome.toxin_avoidance)
        chemotaxis_y = (dnut_y * self.genome.chemotaxis_gain -
                        dtox_y * self.genome.toxin_avoidance)

        # Communication: signal gradients from other cells.
        # Does not dominate; orients when there is resource, alarm, death, or social saturation.
        social_x, social_y = self.communication.z_social_move

        # Integration: neural dominates, chemotaxis and communication as living biases
        total_x = 0.52 * neural_x + 0.30 * chemotaxis_x + 0.18 * social_x
        total_y = 0.52 * neural_y + 0.30 * chemotaxis_y + 0.18 * social_y
        if self.attachment_strength > 0.0:
            # An adhered cell is no longer a free particle. Adhesion does not
            # implement full collective movement, but it makes it costly and
            # rare to break a stable topology by individual displacement.
            drag = np.clip(self.attachment_strength, 0.0, 0.9)
            total_x *= (1.0 - 0.55 * drag)
            total_y *= (1.0 - 0.55 * drag)
            if self.attachment_strength > 0.72 and self.rng.random() > 0.06:
                return

        # Normalize direction
        mag = math.sqrt(total_x**2 + total_y**2) + 1e-9
        if mag > 0.3:
            nx = int(round(total_x / mag))
            ny = int(round(total_y / mag))
        elif self.rng.random() < 0.1:
            nx = self.rng.choice([-1, 0, 1])
            ny = self.rng.choice([-1, 0, 1])
        else:
            return

        new_x = (self.x + nx) % self.world.W
        new_y = (self.y + ny) % self.world.H

        # Movement cost (reduced)
        move_cost = 0.15 + self.genome.motility * 0.1 + self.attachment_strength * 0.25
        actual = self.metabolism.consume_atp(move_cost)
        if actual < move_cost * 0.5:
            return  # insufficient ATP

        if not self.world.is_occupied(new_x, new_y):
            self.world.unregister(self.x, self.y)
            self.x = new_x
            self.y = new_y
            self.world.register(self.x, self.y, self.id)

    # ─────────────────────────────────────────────────────────
    # OFFSPRING (B7 spawn)
    # ─────────────────────────────────────────────────────────

    def _spawn_offspring(self) -> Optional["Cell"]:
        """Creates and positions offspring. Reproduction does not copy full state."""
        child_genome = self.reproduction.build_offspring_genome(
            self.genome, self.neural, self.memory, self.rng
        )
        if child_genome is None:
            return None

        # Search for free adjacent position
        candidates = [(self.x + dx, self.y + dy)
                      for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                      if (dx, dy) != (0, 0)]
        self.rng.shuffle(candidates)

        for cx, cy in candidates:
            cx = cx % self.world.W; cy = cy % self.world.H
            if not self.world.is_occupied(cx, cy):
                # Cost to parent
                self.metabolism.consume_atp(4.0)
                self.metabolism.consume_structural(2.5)
                self.damage_x = min(1.0, self.damage_x + 0.04)

                child = Cell(cx, cy, self.world, child_genome,
                             rng=random.Random(self.rng.randint(0, 2**31)),
                             developing=True)
                child._generation = self._generation + 1
                child._parent_cell_type = self.cell_type  # Feature G: epigenetic bias
                return child

        return None

    # ─────────────────────────────────────────────────────────
    # DEATH CHECK
    # ─────────────────────────────────────────────────────────

    def _check_death(self):
        """
        M1: Metabolic death
        M2: Structural death
        M3: Organizational death
        """
        # M1: critically low energy for sustained period
        if self.metabolism.is_starving:
            self._starving_ticks += 1
            if self._starving_ticks > 15:
                self._die("M1_metabolic")
                return
        else:
            self._starving_ticks = max(0, self._starving_ticks - 1)

        # M1b: terminal toxic waste
        if self.metabolism.w_waste > self.metabolism.w_waste_cap * 0.90:
            self._die("M1_intoxication")
            return

        # M2: structural boundary collapse
        if self.boundary.is_dead:
            self._die("M2_structural")
            return

        # M2b: irrecoverable systemic damage
        if self.damage_x > 0.95:
            self._die("M2_damage")
            return

        # M3: organizational death (identity collapsed sustainedly)
        if self.identity.is_organizationally_dead:
            self._die("M3_organizational")

    def _die(self, cause: str):
        self.alive = False
        self.death_cause = cause
        self.phase = LifePhase.DEAD
        # necrosignal: death also informs the environment.
        if hasattr(self.world, "deposit_signal"):
            self.world.deposit_signal(self.x, self.y, "death_trace", 35.0)
        self.world.unregister(self.x, self.y)
        # returns some nutrients to the environment (matter is not destroyed)
        self.world.deposit_nutrient(self.x, self.y,
            self.metabolism.r_raw * 0.4 + self.metabolism.m_struct * 0.2)
        self._log(f"DEAD:{cause}")

    # ─────────────────────────────────────────────────────────
    # PHASE AND LOGS
    # ─────────────────────────────────────────────────────────

    def _update_phase(self):
        if not self.alive:
            return
        I = self.identity.I
        if self.age_ticks < self.genome.development_ticks:
            self.phase = LifePhase.DEVELOPING
        elif self.reproduction._replicating:
            self.phase = LifePhase.REPLICATING
        elif self.homeostasis.g_stress > 0.5 or self.damage_x > 0.4:
            self.phase = LifePhase.STRESSED
        elif self.damage_x > 0.2 or self.boundary.c_integrity < 0.6:
            self.phase = LifePhase.REPAIRING
        elif self.age_ticks > 400 and self.identity.i_structural_coherence < 0.4:
            self.phase = LifePhase.AGING
        elif I < 0.25:
            self.phase = LifePhase.DYING
        else:
            self.phase = LifePhase.ACTIVE

    def _log(self, msg: str):
        entry = f"[t={self.age_ticks}] {msg}"
        self.event_log.append(entry)
        if len(self.event_log) > 50:
            self.event_log.pop(0)

    def _record_metrics(self):
        if self.age_ticks % 10 == 0:
            self.metrics_history.append({
                "t": self.age_ticks,
                "I": round(self.identity.I, 3),
                "atp": round(self.metabolism.atp_fraction, 3),
                "membrane": round(self.boundary.c_integrity, 3),
                "stress": round(self.homeostasis.g_stress, 3),
                "damage": round(self.damage_x, 3),
                "neural_exc": round(self.neural.t_excitation, 3),
                "mem_int": round(self.memory.h_integrity, 3),
                "comm": round(self.communication.z_signal_load, 3)
            })
            if len(self.metrics_history) > 60:
                self.metrics_history.pop(0)

    def get_status(self) -> Dict:
        return {
            "id": self.id,
            "generation": self._generation,
            "age": self.age_ticks,
            "phase": self.phase.value,
            "cell_type": self.cell_type.value,
            "type_commitment": round(self.type_commitment, 3),
            "alive": self.alive,
            "death_cause": self.death_cause,
            "position": {"x": self.x, "y": self.y},
            # State S(t)
            "S": {
                "C": round(self.boundary.c_integrity, 3),
                "A": round(self.metabolism.atp_fraction, 3),
                "M": round(self.metabolism.m_struct, 2),
                "P": round(self.metabolism.p_repair, 2),
                "W": round(self.metabolism.waste_fraction, 3),
                "X": round(self.damage_x, 3),
                "I": round(self.identity.I, 3),
            },
            # Blocks
            "B1_boundary":    self.boundary.to_dict(),
            "B2_metabolism":  self.metabolism.to_dict(),
            "B3_homeostasis": self.homeostasis.to_dict(),
            "B4_neural":      self.neural.to_dict(),
            "B5_memory":      self.memory.to_dict(),
            "B6_repair":      self.repair.to_dict(),
            "B7_reproduction":self.reproduction.to_dict(),
            "B8_communication": self.communication.to_dict(),
            "B9_identity":    self.identity.to_dict(),
            "multicellular": {
                "junction_count": len(self.junction_ids),
                "organism_id": self.organism_id,
                "attachment_strength": round(self.attachment_strength, 3),
                "spike_output": round(self.spike_output, 3),
                "contribution": round(self.contribution_score, 3),
                "provided_support": round(self.provided_support, 3),
                "received_support": round(self.received_support, 3),
                "cheater_score": round(self.cheater_score, 3),
            },
            # History
            "events": self.event_log[-10:],
            "metrics_history": self.metrics_history[-10:]
        }

class Colony:
    def __init__(self, world: SpatialWorld, max_cells: int = 25):
        self.world     = world
        self.max_cells = max_cells
        self.cells:    Dict[str, Cell] = {}
        self.junctions: Dict[str, Junction] = {}
        self.dead_log: List[Dict]      = []
        self.tick_count = 0
        self.rng = random.Random()
        self.lineage: Dict[str, int] = {}  # generation distribution
        self.organism = OrganismState()
        self._junction_pairs: Set[Tuple[str, str, str]] = set()

    def add_cell(self, cell: Cell) -> bool:
        if len(self.cells) >= self.max_cells:
            return False
        self.cells[cell.id] = cell
        return True

    def spawn_primordial(self, n: int = 4):
        sources = self.world.sources
        placed = 0
        for attempt in range(200):
            if placed >= n:
                break
            # Spawn near nutrient sources
            if sources:
                sx, sy = sources[attempt % len(sources)]
                x = sx + self.rng.randint(-3, 3)
                y = sy + self.rng.randint(-3, 3)
            else:
                x = self.rng.randint(5, self.world.W - 5)
                y = self.rng.randint(5, self.world.H - 5)
            x = x % self.world.W; y = y % self.world.H
            if not self.world.is_occupied(x, y):
                g = Genome.create(self.rng)
                cell = Cell(x, y, self.world, g,
                            rng=random.Random(self.rng.randint(0, 2**31)))
                # More generous initial resources for primordial cells
                cell.metabolism.r_raw  = 50.0
                cell.metabolism.a_free = 60.0
                self.cells[cell.id] = cell
                placed += 1

    def spawn_primordial_from_genome(self, n: int, genome: "Genome"):
        """
        Like spawn_primordial but all cells descend from a founding genome.
        Used by EvolutionEngine to seed replacement colonies with the winning genome.
        """
        sources = self.world.sources
        placed = 0
        for attempt in range(200):
            if placed >= n:
                break
            if sources:
                sx, sy = sources[attempt % len(sources)]
                x = sx + self.rng.randint(-3, 3)
                y = sy + self.rng.randint(-3, 3)
            else:
                x = self.rng.randint(5, self.world.W - 5)
                y = self.rng.randint(5, self.world.H - 5)
            x = x % self.world.W; y = y % self.world.H
            if not self.world.is_occupied(x, y):
                g = genome.mutate(self.rng)
                cell = Cell(x, y, self.world, g,
                            rng=random.Random(self.rng.randint(0, 2**31)))
                cell.metabolism.r_raw  = 50.0
                cell.metabolism.a_free = 60.0
                self.cells[cell.id] = cell
                placed += 1

    def tick(self, include_status: bool = True) -> Optional[Dict]:
        self.world.tick()
        self.tick_count += 1
        self._deliver_synaptic_events()
        self._propagate_gap_currents()   # Feature A: GAP electrical coupling
        self._maintain_junctions_and_share()
        self._update_attachment_strengths()

        to_add, to_remove = self._tick_living_cells()
        self._remove_dead_cells(to_remove)
        self._add_cells(to_add)

        self._form_multicellular_links()
        self._differentiate_cells()
        self._queue_synaptic_events()
        self._police_cells()
        self.organism.update(self.cells, self.junctions)
        self._apply_collective_motor_bias()   # Feature B: collective agency
        self._apply_organism_pressure()
        self._inject_neuromodulation()         # Feature H: organism → neuron feedback
        self._try_organism_reproduction()

        if include_status:
            return self.status()
        return None

    def _tick_living_cells(self) -> Tuple[List[Cell], List[str]]:
        to_add: List[Cell] = []
        to_remove: List[str] = []
        for cid, cell in self.cells.copy().items():
            if not cell.alive:
                to_remove.append(cid)
                continue
            offspring = cell.tick()
            if offspring and len(self.cells) + len(to_add) < self.max_cells:
                to_add.append(offspring)
            if not cell.alive:
                to_remove.append(cid)
        return to_add, to_remove

    def _remove_dead_cells(self, to_remove: List[str]):
        for cid in to_remove:
            dead = self.cells.pop(cid, None)
            if dead is None:
                continue
            self._remove_cell_junctions(dead.id)
            self.dead_log.append({
                "id": dead.id,
                "age": dead.age_ticks,
                "cause": dead.death_cause,
                "gen": dead._generation,
                "tick": self.tick_count
            })
            if len(self.dead_log) > 100:
                self.dead_log.pop(0)

    def _add_cells(self, cells: List[Cell]):
        for cell in cells:
            self.cells[cell.id] = cell

    # ─────────────────────────────────────────────────────────
    # MULTICELLULAR LAYER
    # ─────────────────────────────────────────────────────────

    def _junction_key(self, kind: JunctionKind, a: str, b: str) -> Tuple[str, str, str]:
        if kind == JunctionKind.SYNAPTIC:
            return (kind.value, a, b)
        lo, hi = sorted((a, b))
        return (kind.value, lo, hi)

    def _has_junction(self, kind: JunctionKind, a: str, b: str) -> bool:
        return self._junction_key(kind, a, b) in self._junction_pairs

    def _torus_distance(self, a: Cell, b: Cell) -> float:
        dx = abs(a.x - b.x)
        dy = abs(a.y - b.y)
        dx = min(dx, self.world.W - dx)
        dy = min(dy, self.world.H - dy)
        return math.sqrt(dx * dx + dy * dy)

    def _create_junction(self, kind: JunctionKind, a: str, b: str,
                         strength: float, transport: float,
                         conductance: float, weight: float = 0.0) -> Optional[Junction]:
        if a == b or a not in self.cells or b not in self.cells:
            return None
        key = self._junction_key(kind, a, b)
        if key in self._junction_pairs:
            return None
        jid = str(uuid.uuid4())[:8]
        j = Junction(
            id=jid,
            cell_a=a,
            cell_b=b,
            kind=kind,
            strength=float(np.clip(strength, 0.05, 1.0)),
            stability=float(np.clip(strength, 0.05, 1.0)),
            transport_capacity=float(np.clip(transport, 0.0, 1.0)),
            signal_conductance=float(np.clip(conductance, 0.0, 1.0)),
            maintenance_cost=0.012 + 0.025 * float(np.clip(strength, 0.0, 1.0)),
            weight=float(np.clip(weight, -2.0, 2.0)),
            delay=1 + int(self.rng.random() < 0.25),
            last_activity=self.tick_count,
        )
        self.junctions[jid] = j
        self._junction_pairs.add(key)
        self.cells[a].junction_ids.add(jid)
        self.cells[b].junction_ids.add(jid)
        self.cells[a]._log(f"JUNCTION:{kind.value}:{b}")
        self.cells[b]._log(f"JUNCTION:{kind.value}:{a}")
        return j

    def _remove_junction(self, jid: str):
        j = self.junctions.pop(jid, None)
        if j is None:
            return
        self._junction_pairs.discard(self._junction_key(j.kind, j.cell_a, j.cell_b))
        for cid in (j.cell_a, j.cell_b):
            cell = self.cells.get(cid)
            if cell:
                cell.junction_ids.discard(jid)

    def _remove_cell_junctions(self, cell_id: str):
        for jid in tuple(self.junctions.keys()):
            j = self.junctions[jid]
            if j.cell_a == cell_id or j.cell_b == cell_id:
                self._remove_junction(jid)

    def _compatibility(self, a: Cell, b: Cell) -> float:
        gene_delta = (
            abs(a.genome.membrane_strength - b.genome.membrane_strength) +
            abs(a.genome.transport_capacity - b.genome.transport_capacity) +
            abs(a.genome.signal_receptor_sensitivity - b.genome.signal_receptor_sensitivity)
        ) / 3.0
        kinship = 1.0 - min(1.0, gene_delta)
        vitality = 0.5 * (a.identity.I + b.identity.I) * (1.0 - 0.5 * (a.damage_x + b.damage_x))
        phase_ok = 0.85 if LifePhase.DEAD not in (a.phase, b.phase) else 0.0
        return float(np.clip(0.45 * kinship + 0.45 * vitality + 0.10 * phase_ok, 0.0, 1.0))

    def _form_multicellular_links(self):
        if self.tick_count % 3 != 0:
            return
        alive = [c for c in self.cells.values() if c.alive]
        if len(alive) < 2:
            return

        for cell in alive:
            self._try_form_adhesion(cell)

        for j in tuple(self.junctions.values()):
            self._promote_adhesion(j)

    def _try_form_adhesion(self, cell: Cell):
        if len(cell.junction_ids) >= 8:
            return
        neighbors = self.world.get_neighbor_cells(cell.x, cell.y, 1, self.cells)
        self.rng.shuffle(neighbors)
        for nb in neighbors[:4]:
            other = self.cells.get(nb["id"])
            if self._should_create_adhesion(cell, other):
                comp = self._compatibility(cell, other)
                self._create_junction(
                    JunctionKind.ADHESION, cell.id, other.id,
                    strength=0.22 + 0.45 * comp,
                    transport=0.03,
                    conductance=0.08
                )
                return

    def _should_create_adhesion(self, cell: Cell, other: Optional[Cell]) -> bool:
        if other is None or len(other.junction_ids) >= 8:
            return False
        if self._has_junction(JunctionKind.ADHESION, cell.id, other.id):
            return False
        comp = self._compatibility(cell, other)
        crowd = cell.communication.z_received[SIGNAL_IDX["crowding"]]
        return comp > 0.48 and crowd < 0.75 and self.rng.random() < comp * 0.16

    def _promote_adhesion(self, j: Junction):
        if j.kind != JunctionKind.ADHESION or j.age < 6 or j.strength < 0.35:
            return
        a = self.cells.get(j.cell_a)
        b = self.cells.get(j.cell_b)
        if a is None or b is None:
            return
        self._maybe_create_metabolic_link(a, b, j)
        self._maybe_create_synaptic_link(a, b, j)

    def _maybe_create_metabolic_link(self, a: Cell, b: Cell, j: Junction):
        if self._has_junction(JunctionKind.METABOLIC, a.id, b.id):
            return
        need_gap = abs(a.metabolism.atp_fraction - b.metabolism.atp_fraction)
        repair_gap = abs(a.damage_x - b.damage_x)
        if need_gap + repair_gap > 0.18 and self.rng.random() < 0.18:
            self._create_junction(
                JunctionKind.METABOLIC, a.id, b.id,
                strength=j.strength * 0.65,
                transport=0.10 + 0.40 * j.strength,
                conductance=0.18
            )

    def _maybe_create_synaptic_link(self, a: Cell, b: Cell, j: Junction):
        if self._has_junction(JunctionKind.SYNAPTIC, a.id, b.id):
            return
        neural_types = (CellType.NEURON, CellType.SENSORY, CellType.MOTOR)
        if a.cell_type not in neural_types and b.cell_type not in neural_types:
            return
        if self.rng.random() >= 0.12:
            return
        pre, post = (a, b) if a.id < b.id else (b, a)
        syn = self._create_junction(
            JunctionKind.SYNAPTIC, pre.id, post.id,
            strength=j.strength * 0.55,
            transport=0.0,
            conductance=0.20 + 0.45 * j.strength,
            weight=0.18 + 0.45 * j.strength
        )
        if syn is not None and self.rng.random() < 0.30:
            syn.receptor_type = "inhibitory"
            syn.weight = -abs(syn.weight)

    def _maintain_junctions_and_share(self):
        for j in tuple(self.junctions.values()):
            pair = self._live_junction_cells(j)
            if pair is None:
                continue
            a, b = pair
            j.age += 1
            if self._remove_if_too_distant(j, a, b):
                continue
            self._pay_junction_maintenance(j, a, b)
            if j.kind in (JunctionKind.METABOLIC, JunctionKind.GAP):
                self._resource_share(a, b, j)
            if self._should_prune_junction(j):
                self._remove_junction(j.id)

    def _live_junction_cells(self, j: Junction) -> Optional[Tuple[Cell, Cell]]:
        a = self.cells.get(j.cell_a)
        b = self.cells.get(j.cell_b)
        if a is None or b is None or not a.alive or not b.alive:
            self._remove_junction(j.id)
            return None
        return a, b

    def _remove_if_too_distant(self, j: Junction, a: Cell, b: Cell) -> bool:
        distance = self._torus_distance(a, b)
        max_distance = 4.0 if j.kind == JunctionKind.SYNAPTIC else 2.0
        if distance <= max_distance:
            return False
        j.strength = max(0.0, j.strength - 0.08 * (distance - max_distance))
        j.stability = max(0.0, j.stability - 0.04 * (distance - max_distance))
        if distance <= max_distance * 1.8:
            return False
        self._remove_junction(j.id)
        return True

    def _pay_junction_maintenance(self, j: Junction, a: Cell, b: Cell):
        type_discount = 0.75 if CellType.BOUNDARY in (a.cell_type, b.cell_type) else 1.0
        due = j.maintenance_cost * type_discount
        paid = a.metabolism.consume_atp(due * 0.5) + b.metabolism.consume_atp(due * 0.5)
        if paid + 1e-9 < due * 0.65:
            j.stability = max(0.0, j.stability - 0.035)
            j.strength = max(0.0, j.strength - 0.025)
            return
        j.stability = min(1.0, j.stability + 0.006)
        j.strength = min(1.0, j.strength + 0.004 * j.stability)

    def _should_prune_junction(self, j: Junction) -> bool:
        idle = self.tick_count - j.last_activity
        if j.kind == JunctionKind.SYNAPTIC and idle > 80:
            j.prune_score += 0.015
        return j.strength < 0.04 or j.stability < 0.03 or j.prune_score > 1.0

    def _resource_share(self, a: Cell, b: Cell, j: Junction):
        cap = j.transport_capacity * j.strength
        if cap <= 0:
            return
        self._share_resource_gap(a, b, j, cap, "a_free", "a_free_cap", "atp_fraction", 0.16, 0.35, 2.2, 1.0)
        self._share_resource_gap(a, b, j, cap, "r_raw", "r_raw_cap", "r_raw", 16.0, None, 1.6, 0.45)
        self._share_repair_material(a, b, j, cap)
        self._share_waste(a, b, j, cap)

    def _move_resource_between(self, attr: str, cap_attr: str, donor: Cell, receiver: Cell,
                               amount: float, support_weight: float, j: Junction):
        amount = max(0.0, amount)
        available = getattr(donor.metabolism, attr)
        actual = min(available, amount)
        if actual <= 1e-9:
            return
        receiver_cap = getattr(receiver.metabolism, cap_attr)
        accepted = min(actual, max(0.0, receiver_cap - getattr(receiver.metabolism, attr)))
        if accepted <= 1e-9:
            return
        setattr(donor.metabolism, attr, available - accepted)
        setattr(receiver.metabolism, attr, getattr(receiver.metabolism, attr) + accepted)
        donor.provided_support += accepted * support_weight
        receiver.received_support += accepted * support_weight
        donor.contribution_score = min(10.0, donor.contribution_score + accepted * support_weight * 0.02)
        j.last_activity = self.tick_count

    def _share_resource_gap(self, a: Cell, b: Cell, j: Junction, cap: float,
                            attr: str, cap_attr: str, metric: str, gap: float,
                            donor_floor: Optional[float], amount_mult: float, support_weight: float):
        a_value = getattr(a.metabolism, metric)
        b_value = getattr(b.metabolism, metric)
        if a_value > b_value + gap and (donor_floor is None or a_value > donor_floor):
            self._move_resource_between(attr, cap_attr, a, b, cap * amount_mult, support_weight, j)
        elif b_value > a_value + gap and (donor_floor is None or b_value > donor_floor):
            self._move_resource_between(attr, cap_attr, b, a, cap * amount_mult, support_weight, j)

    def _share_repair_material(self, a: Cell, b: Cell, j: Junction, cap: float):
        if a.metabolism.p_repair > b.metabolism.p_repair + 8 and b.damage_x > a.damage_x:
            self._move_resource_between("p_repair", "p_repair_cap", a, b, cap * 1.1, 1.2, j)
        elif b.metabolism.p_repair > a.metabolism.p_repair + 8 and a.damage_x > b.damage_x:
            self._move_resource_between("p_repair", "p_repair_cap", b, a, cap * 1.1, 1.2, j)

    def _share_waste(self, a: Cell, b: Cell, j: Junction, cap: float):
        if a.metabolism.waste_fraction > b.metabolism.waste_fraction + 0.18 and b.metabolism.waste_fraction < 0.55:
            self._move_waste(a, b, j, cap)
        elif b.metabolism.waste_fraction > a.metabolism.waste_fraction + 0.18 and a.metabolism.waste_fraction < 0.55:
            self._move_waste(b, a, j, cap)

    def _move_waste(self, donor: Cell, receiver: Cell, j: Junction, cap: float):
        actual = min(
            donor.metabolism.w_waste, cap * 0.9,
            max(0.0, receiver.metabolism.w_waste_cap - receiver.metabolism.w_waste)
        )
        donor.metabolism.w_waste -= actual
        receiver.metabolism.w_waste += actual
        receiver.provided_support += actual * 0.6
        donor.received_support += actual * 0.6
        j.last_activity = self.tick_count

    def _update_attachment_strengths(self):
        for c in self.cells.values():
            strengths = [
                self.junctions[jid].strength for jid in c.junction_ids
                if jid in self.junctions and self.junctions[jid].kind == JunctionKind.ADHESION
            ]
            c.attachment_strength = float(np.clip(sum(strengths) / 3.0, 0.0, 1.0))

    def _propagate_gap_currents(self):
        """Feature A: propagates electrical current through GAP junctions before the cell tick.

        GAP junctions transmit electrical excitation instantaneously
        (same tick), unlike chemical synapses (1-tick delay).
        This enables the formation of real neural circuits between cells.
        """
        for j in self.junctions.values():
            if j.kind != JunctionKind.GAP:
                continue
            a = self.cells.get(j.cell_a)
            b = self.cells.get(j.cell_b)
            if a is not None and b is not None and a.alive and b.alive:
                if a.spike_output > 0.0:
                    b.neural.apply_gap_current(j.signal_conductance * a.spike_output)
                if b.spike_output > 0.0:
                    a.neural.apply_gap_current(j.signal_conductance * b.spike_output)

    def _apply_collective_motor_bias(self):
        """Feature B: applies the collective motor output of the organism to SENSORY cells.

        The aggregated neural state of NEURON/MOTOR cells in the organism biases
        the movement of SENSORY cells, materializing collective agency.
        Only acts when there is sufficient neural coordination and identity cohesion.
        """
        for org in self.organism.organisms:
            if org.collective_identity <= 0.35 or len(org.member_cell_ids) < 3:
                continue
            scale = org.neural_coordination
            if scale < 1e-6:
                continue
            dx_bias, dy_bias, _ = org.collective_motor_output
            for cid in org.member_cell_ids:
                c = self.cells.get(cid)
                if c is None or not c.alive or c.cell_type != CellType.SENSORY:
                    continue
                c.communication.z_social_move = np.clip(
                    c.communication.z_social_move +
                    np.array([dx_bias, dy_bias]) * scale * 0.25,
                    -1.0, 1.0
                )

    def _deliver_synaptic_events(self):
        for j in tuple(self.junctions.values()):
            if j.kind != JunctionKind.SYNAPTIC:
                continue
            post = self.cells.get(j.cell_b)
            if post is None or not post.alive:
                self._remove_junction(j.id)
                continue
            j.event_queue = self._deliver_event_queue(j, post)

    def _deliver_event_queue(self, j: Junction, post: Cell) -> List[Tuple[int, float]]:
        pending: List[Tuple[int, float]] = []
        for deliver_tick, amount in j.event_queue:
            if deliver_tick > self.tick_count:
                pending.append((deliver_tick, amount))
                continue
            sign = -1.0 if j.receptor_type == "inhibitory" else 1.0
            # Feature F: homeostatic scale keeps post-synaptic cell near its target rate
            current = sign * amount * j.weight * j.signal_conductance * post.neural.homeostatic_scale
            post.synaptic_input += np.array([current, current * 0.45, current * 0.25, -current * 0.15])
            j.last_activity = self.tick_count
        return pending

    def _queue_synaptic_events(self):
        for j in tuple(self.junctions.values()):
            if j.kind != JunctionKind.SYNAPTIC:
                continue
            pair = self._live_junction_cells(j)
            if pair is None:
                continue
            pre, post = pair
            self._queue_pre_spike(j, pre)
            self._record_post_spike(j, post)
            self._apply_synaptic_plasticity(j, pre, post)

    def _queue_pre_spike(self, j: Junction, pre: Cell):
        if pre.spike_output <= 0.05:
            return
        pre.last_spike_tick = self.tick_count
        j.last_pre_tick = self.tick_count
        j.event_queue.append((self.tick_count + j.delay, pre.spike_output))
        j.prune_score = max(0.0, j.prune_score - 0.08)

    def _record_post_spike(self, j: Junction, post: Cell):
        if post.spike_output > 0.05:
            post.last_spike_tick = self.tick_count
            j.last_post_tick = self.tick_count

    def _apply_synaptic_plasticity(self, j: Junction, pre: Cell, post: Cell):
        if j.last_pre_tick is None or j.last_post_tick is None:
            return
        dt = j.last_post_tick - j.last_pre_tick
        if not -8 <= dt <= 8:
            return
        org_bonus = self._synaptic_org_bonus(pre, post)
        if dt >= 0:
            j.weight += (0.012 + 0.004 * org_bonus) * math.exp(-dt / 4.0)
        else:
            j.weight -= (0.010 + 0.003 * max(0.0, -org_bonus)) * math.exp(dt / 4.0)
        self._update_synaptic_utility(j, org_bonus)

    def _synaptic_org_bonus(self, pre: Cell, post: Cell) -> float:
        org = self.organism.get(post.organism_id)
        if org is None or pre.organism_id != post.organism_id:
            return 0.0
        functional_loop = (
            pre.cell_type in (CellType.SENSORY, CellType.NEURON) and
            post.cell_type in (CellType.NEURON, CellType.MOTOR, CellType.REPAIR, CellType.METABOLIC)
        )
        viability = (
            org.collective_identity -
            0.35 * org.collective_damage -
            0.25 * org.collective_energy_pressure
        )
        return float(np.clip(viability, -0.35, 0.65) * (1.35 if functional_loop else 0.65))

    def _update_synaptic_utility(self, j: Junction, org_bonus: float):
        j.utility_trace = float(np.clip(0.94 * j.utility_trace + 0.06 * org_bonus, -1.0, 1.0))
        j.weight = float(np.clip(j.weight, -1.5, 1.8))
        if j.utility_trace < -0.25:
            j.prune_score += 0.010
        else:
            j.prune_score = max(0.0, j.prune_score - 0.012 * max(0.0, j.utility_trace))
        j.strength = float(np.clip(
            j.strength + 0.003 * abs(j.weight) + 0.002 * max(0.0, j.utility_trace),
            0.0, 1.0
        ))
        j.last_activity = self.tick_count

    def _differentiate_cells(self):
        if self.tick_count % 5 != 0:
            return
        alive_cells = [c for c in self.cells.values() if c.alive]
        type_counts, organism_type_counts, organism_sizes = self._differentiation_counts(alive_cells)
        role_floor = max(1, int(len(alive_cells) * 0.06))
        for cell in alive_cells:
            scores = self._differentiation_scores(
                cell, alive_cells, type_counts, organism_type_counts,
                organism_sizes, role_floor
            )
            self._apply_differentiation(cell, scores)

    def _differentiation_counts(self, alive_cells: List[Cell]):
        type_counts: Dict[CellType, int] = {}
        organism_type_counts: Dict[str, Dict[CellType, int]] = {}
        organism_sizes: Dict[str, int] = {}
        for c in alive_cells:
            type_counts[c.cell_type] = type_counts.get(c.cell_type, 0) + 1
            if c.organism_id:
                org_counts = organism_type_counts.setdefault(c.organism_id, {})
                org_counts[c.cell_type] = org_counts.get(c.cell_type, 0) + 1
                organism_sizes[c.organism_id] = organism_sizes.get(c.organism_id, 0) + 1
        return type_counts, organism_type_counts, organism_sizes

    def _role_scarcity(self, target_type: CellType, cell: Cell,
                       alive_cells: List[Cell], type_counts: Dict[CellType, int],
                       organism_type_counts: Dict[str, Dict[CellType, int]],
                       organism_sizes: Dict[str, int], role_floor: int) -> float:
        if cell.organism_id and cell.organism_id in organism_type_counts:
            org_size = organism_sizes.get(cell.organism_id, len(alive_cells))
            org_floor = max(1, int(org_size * 0.12))
            org_count = organism_type_counts[cell.organism_id].get(target_type, 0)
            return max(0.0, (org_floor - org_count) / max(1.0, org_floor))
        return max(0.0, (role_floor - type_counts.get(target_type, 0)) / max(1.0, role_floor))

    def _differentiation_scores(self, c: Cell, alive_cells: List[Cell],
                                type_counts: Dict[CellType, int],
                                organism_type_counts: Dict[str, Dict[CellType, int]],
                                organism_sizes: Dict[str, int],
                                role_floor: int) -> Dict[CellType, float]:
        neighbors = self.world.get_neighbor_cells(c.x, c.y, 2, self.cells)
        local_signals = self.world.sample_signals(c.x, c.y)
        morph_bias = self._morphogen_bias(c)
        org_counts, org_identity, org_size = self._organism_role_context(c, alive_cells, organism_type_counts, organism_sizes)
        scarcity = lambda t: self._role_scarcity(t, c, alive_cells, type_counts, organism_type_counts, organism_sizes, role_floor)
        needs = self._organism_role_needs(org_counts, org_identity, org_size)
        adhesion_count = self._junction_count(c, JunctionKind.ADHESION)
        syn_count = self._junction_count(c, JunctionKind.SYNAPTIC)
        neighbor_damage = np.mean([n["damage"] for n in neighbors]) if neighbors else 0.0
        scores = {
            CellType.STEM: 0.15 if c.age_ticks < c.genome.development_ticks * 2 else 0.02,
            CellType.BOUNDARY: (0.25 if adhesion_count > 0 else 0.0) +
                               max(0.0, 4 - len(neighbors)) * 0.06 +
                               c.boundary.c_integrity * 0.10 + scarcity(CellType.BOUNDARY) * 0.20 +
                               float(morph_bias[0]),
            CellType.METABOLIC: c.metabolism.atp_fraction * 0.45 + min(1.0, c.metabolism.r_raw / 80.0) * 0.25 +
                                scarcity(CellType.METABOLIC) * 0.52 + float(morph_bias[2]),
            CellType.REPAIR: neighbor_damage * 0.75 + c.homeostasis.g_damage_error * 0.35 +
                             scarcity(CellType.REPAIR) * 0.50 + float(morph_bias[3]),
            CellType.SIGNALING: c.communication.z_signal_load * 0.80 + local_signals[SIGNAL_IDX["crowding"]] / 100.0 +
                                scarcity(CellType.SIGNALING) * 0.46,
            CellType.NEURON: c.neural.t_excitation * 0.65 + syn_count * 0.18 + c.memory.h_integrity * 0.10 +
                             scarcity(CellType.NEURON) * 0.36 + needs["neural"] + float(morph_bias[1]),
            CellType.SENSORY: c.communication.z_coherence * 0.18 + np.mean(local_signals > 3.0) * 0.28 + needs["sensory"],
            CellType.MOTOR: c.genome.motility * 0.35 + abs(c.neural.t_action_bias[0]) * 0.20 +
                            abs(c.neural.t_action_bias[1]) * 0.20 + needs["motor"],
            CellType.GERMLINE: c.reproduction.r_maturity * 0.65 + c.identity.I * 0.20 -
                               c.damage_x * 0.4 + needs["germline"],
            CellType.POLICING: c.cheater_score * 0.15 + max(0.0, self.organism.shared_stress - 0.35) * 0.65,
        }
        # Feature G: epigenetic cell fate bias from parent's cell type.
        # Young cells (low type_commitment) inherit a push toward their parent's
        # lineage, enabling clonal expansion of specialized tissue.
        if c._parent_cell_type is not None and c.type_commitment < 0.60:
            bias = 0.15 * (1.0 - c.type_commitment / 0.60)
            scores[c._parent_cell_type] = scores[c._parent_cell_type] + bias
        return scores

    def _morphogen_bias(self, cell: Cell) -> np.ndarray:
        morph_a, morph_b = self.world.sample_morphogens(cell.x, cell.y)
        morph_resp = cell.genome.morphogen_response()
        return np.clip(morph_a * morph_resp[0] + morph_b * morph_resp[1], -0.30, 0.30)

    def _organism_role_context(self, cell: Cell, alive_cells: List[Cell],
                               organism_type_counts: Dict[str, Dict[CellType, int]],
                               organism_sizes: Dict[str, int]):
        org = self.organism.get(cell.organism_id)
        org_counts = organism_type_counts.get(cell.organism_id, {}) if cell.organism_id else {}
        org_size = organism_sizes.get(cell.organism_id, len(alive_cells)) if cell.organism_id else len(alive_cells)
        org_identity = org.collective_identity if org is not None else self.organism.collective_identity
        return org_counts, org_identity, org_size

    def _organism_role_needs(self, org_counts: Dict[CellType, int], org_identity: float, org_size: int) -> Dict[str, float]:
        return {
            "neural": 0.26 if org_identity > 0.42 and org_counts.get(CellType.NEURON, 0) == 0 else 0.0,
            "sensory": 0.16 if org_identity > 0.45 and org_counts.get(CellType.SENSORY, 0) == 0 else 0.0,
            "motor": 0.16 if org_identity > 0.45 and org_counts.get(CellType.MOTOR, 0) == 0 else 0.0,
            "germline": 0.42 if org_identity > 0.45 and org_size >= 6 and
            org_counts.get(CellType.GERMLINE, 0) == 0 else 0.0,
        }

    def _junction_count(self, cell: Cell, kind: JunctionKind) -> int:
        return sum(1 for jid in cell.junction_ids
                   if jid in self.junctions and self.junctions[jid].kind == kind)

    def _apply_differentiation(self, cell: Cell, scores: Dict[CellType, float]):
        target = max(scores, key=scores.get)
        current_score = scores.get(cell.cell_type, 0.0)
        if target == cell.cell_type or scores[target] <= current_score + 0.12 or cell.type_commitment >= 0.82:
            cell.type_commitment = min(1.0, cell.type_commitment + 0.025)
            return
        cost = 0.18 + 0.25 * cell.type_commitment
        if cell.metabolism.consume_atp(cost) >= cost * 0.7:
            old = cell.cell_type
            cell.cell_type = target
            cell.type_commitment = min(1.0, cell.type_commitment + 0.22)
            cell._log(f"DIFFERENTIATED:{old.value}->{target.value}")

    def _police_cells(self):
        for c in self.cells.values():
            self._update_policing_score(c)
            if c.cheater_score > 0.72:
                self._penalize_cheater(c)

    def _update_policing_score(self, cell: Cell):
        cell.received_support *= 0.94
        cell.provided_support *= 0.94
        cell.contribution_score *= 0.995
        imbalance = cell.received_support - cell.provided_support * 1.8
        if imbalance > 1.0 and cell.metabolism.atp_fraction > 0.38 and len(cell.junction_ids) >= 2:
            cell.cheater_score = min(1.0, cell.cheater_score + 0.035 * imbalance)
            return
        cell.cheater_score = max(0.0, cell.cheater_score - 0.015)

    def _penalize_cheater(self, cell: Cell):
        cell.reproduction.r_maturity = max(0.0, cell.reproduction.r_maturity - 0.015)
        for jid in tuple(cell.junction_ids):
            j = self.junctions.get(jid)
            if j:
                j.strength = max(0.0, j.strength - 0.012)
        if cell.cell_type == CellType.POLICING:
            cell.cheater_score *= 0.8

    def _apply_organism_pressure(self):
        for org in self.organism.organisms:
            if org.collective_identity <= 0.0:
                continue
            for cid in org.member_cell_ids:
                c = self.cells.get(cid)
                if c is not None and c.alive:
                    self._apply_pressure_to_cell(org, c)

    def _apply_pressure_to_cell(self, org: OrganismInstance, cell: Cell):
        if org.shared_stress > 0.38:
            cell.homeostasis.p_repair = min(0.85, cell.homeostasis.p_repair + 0.015)
            cell.homeostasis.p_reproduction = max(0.02, cell.homeostasis.p_reproduction - 0.010)
        if len(org.member_cell_ids) >= 3 and org.collective_identity < 0.22:
            cell.damage_x = min(1.0, cell.damage_x + 0.002)
            cell.reproduction.r_maturity = max(0.0, cell.reproduction.r_maturity - 0.006)
        self._reward_role_pressure(org, cell)

    def _reward_role_pressure(self, org: OrganismInstance, cell: Cell):
        if org.boundary_closure < 0.25 and cell.cell_type == CellType.BOUNDARY:
            cell.contribution_score = min(10.0, cell.contribution_score + 0.004)
        if org.neural_coordination > 0.16 and cell.cell_type in (CellType.SENSORY, CellType.NEURON, CellType.MOTOR):
            cell.contribution_score = min(10.0, cell.contribution_score + 0.003)
        if org.collective_identity > 0.52 and cell.cell_type in (CellType.BOUNDARY, CellType.REPAIR):
            cell.contribution_score = min(10.0, cell.contribution_score + 0.002)

    def _inject_neuromodulation(self):
        """Feature H: organism-level neuromodulatory feedback.

        When the organism has sufficient neural coordination and collective
        identity, inject a top-down bias into member NEURON and SENSORY cells'
        synaptic_input. This closes the causal loop: organism-level neural state
        modulates individual cell excitability, enabling true hierarchical control.
        Analogous to neuromodulatory systems (dopamine, norepinephrine) in real brains.
        """
        for org in self.organism.organisms:
            if org.neural_coordination > 0.12 and org.collective_identity > 0.35:
                amp = org.neural_coordination * org.collective_identity
                motor_x, motor_y, motor_cap = org.collective_motor_output
                stress_bias = -org.shared_stress * 0.12 * amp
                for cid in org.member_cell_ids:
                    c = self.cells.get(cid)
                    if c is None or not c.alive or c.cell_type not in (CellType.NEURON, CellType.SENSORY):
                        continue
                    modulation = np.array([
                        stress_bias + motor_x * amp * 0.10,
                        stress_bias + motor_y * amp * 0.10,
                        motor_cap * amp * 0.08,
                        -org.shared_stress * 0.05 * amp,
                    ], dtype=np.float64)
                    c.synaptic_input = np.clip(c.synaptic_input + modulation, -1.0, 1.0)

    def _free_positions_near(self, x: int, y: int, radius: int, limit: int) -> List[Tuple[int, int]]:
        candidates = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % self.world.W
                ny = (y + dy) % self.world.H
                if not self.world.is_occupied(nx, ny):
                    candidates.append((nx, ny))
        self.rng.shuffle(candidates)
        return candidates[:limit]

    def _try_organism_reproduction(self):
        if self.tick_count % 40 != 0:
            return
        if len(self.cells) + 3 > self.max_cells:
            return
        candidates = [org for org in self.organism.organisms if self._is_reproductive_org(org)]
        if not candidates:
            return
        parent_org = max(candidates, key=lambda o: o.evolutionary_score() + o.reproduction_pressure)
        seed_count = self._organism_seed_count(parent_org)
        if len(self.cells) + seed_count > self.max_cells:
            return

        germline = self._viable_germline_cells(parent_org)
        if not germline:
            return
        parent = max(germline, key=lambda c: c.identity.I + c.metabolism.atp_fraction)
        positions = self._free_positions_near(parent.x, parent.y, 2, seed_count)
        if len(positions) < seed_count:
            return

        child_genome = self._organism_child_genome(parent, germline)
        if child_genome is None:
            parent._log("ORGANISM_REPRO_ABORTED:heredity")
            return

        if not self._pay_organism_reproduction_cost(parent, seed_count):
            return

        created = self._create_seed_cluster(parent, child_genome, positions, seed_count)
        self._link_seed_cluster(created)
        parent.reproduction.r_maturity = max(0.0, parent.reproduction.r_maturity - 0.35)
        parent._log(f"ORGANISM_REPRODUCTION:{parent_org.organism_id}:seed_cluster")
        self.organism.update(self.cells, self.junctions)

    def _is_reproductive_org(self, org: OrganismInstance) -> bool:
        return (
            org.collective_identity >= 0.58 and
            org.role_coverage >= 0.8 and
            org.reproduction_pressure >= 0.07 and
            len(org.member_cell_ids) >= 3
        )

    def _organism_seed_count(self, org: OrganismInstance) -> int:
        if org.neural_coordination > 0.18:
            return 5
        return 4 if org.boundary_closure > 0.35 else 3

    def _viable_germline_cells(self, org: OrganismInstance) -> List[Cell]:
        cells = [self.cells[cid] for cid in org.member_cell_ids if cid in self.cells]
        return [
            c for c in cells
            if c.alive and c.cell_type == CellType.GERMLINE and
            c.reproduction.r_maturity > 0.7 and c.identity.I > 0.55 and
            c.metabolism.a_free > 20 and c.metabolism.m_struct > 25
        ]

    def _organism_child_genome(self, parent: Cell, germline: List[Cell]) -> Optional[Genome]:
        if len(germline) >= 2:
            second = sorted(germline, key=lambda c: c.identity.I)[-2]
            base_genome = parent.genome.recombine(second.genome, parent.rng)
        else:
            base_genome = parent.genome
        return parent.reproduction.build_offspring_genome(
            base_genome, parent.neural, parent.memory, parent.rng
        )

    def _pay_organism_reproduction_cost(self, parent: Cell, seed_count: int) -> bool:
        atp_cost = 10.0 + 2.0 * seed_count
        struct_cost = 7.0 + 1.5 * seed_count
        return (
            parent.metabolism.consume_atp(atp_cost) >= atp_cost * 0.8 and
            parent.metabolism.consume_structural(struct_cost) >= struct_cost * 0.8
        )

    def _create_seed_cluster(self, parent: Cell, child_genome: Genome,
                             positions: List[Tuple[int, int]], seed_count: int) -> List[Cell]:
        seed_types = [CellType.STEM, CellType.BOUNDARY, CellType.METABOLIC, CellType.REPAIR, CellType.NEURON][:seed_count]
        created: List[Cell] = []
        for (px, py), target_type in zip(positions, seed_types):
            g = child_genome.mutate(parent.rng)
            child = Cell(px, py, self.world, g,
                         rng=random.Random(parent.rng.randint(0, 2**31)),
                         developing=True)
            child._generation = parent._generation + 1
            child.cell_type = target_type
            child.type_commitment = 0.45
            child.metabolism.a_free = min(child.metabolism.a_free_cap, 32.0)
            child.metabolism.r_raw = min(child.metabolism.r_raw_cap, 36.0)
            self.cells[child.id] = child
            created.append(child)
        return created

    def _link_seed_cluster(self, created: List[Cell]):
        if len(created) >= 2:
            self._create_junction(JunctionKind.ADHESION, created[0].id, created[1].id, 0.55, 0.04, 0.10)
        if len(created) >= 3:
            self._create_junction(JunctionKind.ADHESION, created[1].id, created[2].id, 0.55, 0.04, 0.10)
            self._create_junction(JunctionKind.METABOLIC, created[0].id, created[2].id, 0.42, 0.24, 0.12)
        if len(created) >= 4:
            self._create_junction(JunctionKind.ADHESION, created[1].id, created[3].id, 0.44, 0.04, 0.10)
            self._create_junction(JunctionKind.METABOLIC, created[2].id, created[3].id, 0.34, 0.20, 0.12)
        if len(created) >= 5:
            syn = self._create_junction(JunctionKind.SYNAPTIC, created[4].id, created[3].id, 0.36, 0.0, 0.24, 0.24)
            if syn is not None and self.rng.random() < 0.25:
                syn.receptor_type = "inhibitory"
                syn.weight = -abs(syn.weight)

    def status(self) -> Dict:
        alive = [c for c in self.cells.values() if c.alive]
        phases = {}
        for c in alive:
            phases[c.phase.value] = phases.get(c.phase.value, 0) + 1

        gen_dist = {}
        for c in alive:
            g = str(c._generation)
            gen_dist[g] = gen_dist.get(g, 0) + 1

        type_dist = {}
        for c in alive:
            t = c.cell_type.value
            type_dist[t] = type_dist.get(t, 0) + 1

        avg_identity = sum(c.identity.I for c in alive) / (len(alive) + 1e-9)

        return {
            "tick": self.tick_count,
            "alive": len(alive),
            "max": self.max_cells,
            "phases": phases,
            "generations": gen_dist,
            "cell_types": type_dist,
            "avg_identity_I": round(avg_identity, 3),
            "organism": self.organism.to_dict(),
            "organisms": [o.to_dict() for o in self.organism.organisms[:12]],
            "junction_count": len(self.junctions),
            "junctions": [j.to_dict() for j in tuple(self.junctions.values())[:80]],
            "recent_deaths": self.dead_log[-5:],
            "cells": [c.get_status() for c in alive]
        }


# ─────────────────────────────────────────────────────────────
# COLONY WORKER — one colony per process/thread
# ─────────────────────────────────────────────────────────────

class _ColonyWorker:
    """Owns one colony. Instantiated inside each worker process/thread."""

    def __init__(self, init_params: Dict) -> None:
        col_rng = random.Random(init_params["col_rng_seed"])
        per_colony_cells = init_params["per_colony_cells"]
        scale = derive_simulation_scale(per_colony_cells)
        self._scale = scale
        world = SpatialWorld(
            width=scale.world_width, height=scale.world_height,
            n_sources=scale.n_sources, rng=col_rng, source_strength=8.0,
            perturbation_interval=init_params["perturbation_interval"],
            perturbation_strength=init_params["perturbation_strength"],
        )
        for _ in range(scale.prewarm_ticks):
            world.tick()
        self.colony = Colony(world, max_cells=scale.max_cells)
        self.colony.rng = col_rng
        self._founding_genome = init_params["founding_genome"]
        self.colony.spawn_primordial_from_genome(scale.primordial_cells, self._founding_genome)

    def tick(self, global_signal: Optional[np.ndarray] = None) -> Tuple[float, bool, int, str, np.ndarray]:
        if global_signal is not None:
            self._apply_global_signal(global_signal)
        self.colony.tick(include_status=False)
        alive_cells = [c for c in self.colony.cells.values() if c.alive]
        fitness = self._compute_fitness(alive_cells)
        if alive_cells:
            xs = [c.x for c in alive_cells]
            ys = [c.y for c in alive_cells]
            signal_summary = self.colony.world.signals[:, ys, xs].mean(axis=1)
        else:
            signal_summary = np.zeros(N_SIGNAL_CHANNELS, dtype=np.float64)
        return fitness, len(alive_cells) == 0, len(alive_cells), self.colony.organism.development_stage, signal_summary

    def _apply_global_signal(self, global_signal: np.ndarray) -> None:
        world = self.colony.world
        leak = GLOBAL_SIGNAL_LEAKAGE * global_signal
        positions = list(world.occupied.keys())
        if not positions:
            cx, cy = world.W // 2, world.H // 2
            for ch in range(N_SIGNAL_CHANNELS):
                if leak[ch] > 0.0:
                    world.deposit_signal(cx, cy, ch, float(leak[ch]))
        else:
            per_cell = leak / len(positions)
            nonzero = [(ch, float(per_cell[ch]))
                       for ch in range(N_SIGNAL_CHANNELS) if per_cell[ch] > 0.0]
            for (x, y) in positions:
                for ch, amount in nonzero:
                    world.deposit_signal(x, y, ch, amount)

    def get_emigrants(self, max_count: int) -> List[Dict]:
        world = self.colony.world
        W, H = world.W, world.H
        candidates = [
            c for c in self.colony.cells.values()
            if c.alive and (c.x <= 1 or c.x >= W - 2 or c.y <= 1 or c.y >= H - 2)
        ]
        if not candidates:
            return []
        candidates.sort(key=lambda c: c.genome.motility * c.identity.I, reverse=True)
        emigrants: List[Dict] = []
        for cell in candidates[:max_count]:
            emigrants.append({
                "genome": cell.genome,
                "generation": cell._generation,
                "cell_type_value": cell.cell_type.value,
            })
            cell.alive = False
            cell.death_cause = "emigration"
        return emigrants

    def receive_immigrant(self, emigrant: Dict) -> None:
        if len(self.colony.cells) >= self.colony.max_cells:
            return
        world = self.colony.world
        W, H = world.W, world.H
        free_boundary = [
            (x, y)
            for x in range(W) for y in range(H)
            if (x <= 1 or x >= W - 2 or y <= 1 or y >= H - 2)
            and not world.is_occupied(x, y)
        ]
        if not free_boundary:
            return
        x, y = self.colony.rng.choice(free_boundary)
        cell = Cell(x, y, world, emigrant["genome"],
                    rng=random.Random(self.colony.rng.randint(0, 2**31)),
                    developing=True)
        cell._generation = emigrant["generation"]
        try:
            cell.cell_type = CellType(emigrant["cell_type_value"])
        except ValueError:
            cell.cell_type = CellType.STEM
        self.colony.cells[cell.id] = cell

    def _compute_fitness(self, alive_cells: List) -> float:
        if not alive_cells:
            return 0.0
        mean_gen = 1.0 + sum(c._generation for c in alive_cells) / len(alive_cells)
        mean_identity = sum(c.identity.I for c in alive_cells) / len(alive_cells)
        organism_scores = [o.evolutionary_score() for o in self.colony.organism.organisms]
        best_body = max(organism_scores) if organism_scores else 0.0
        viable_bodies = sum(
            1 for o in self.colony.organism.organisms
            if o.development_stage in ("proto_tissue", "integrated", "integrated_body")
        )
        body_diversity = min(1.0, viable_bodies / 4.0)
        stage_mult = {
            "solitary": 0.45, "aggregate": 0.75, "proto_tissue": 1.25,
            "integrated": 1.8, "integrated_body": 2.3, "extinct": 0.0,
        }.get(self.colony.organism.development_stage, 0.5)
        return float(mean_gen * mean_identity * stage_mult * (0.65 + best_body + 0.25 * body_diversity))

    def get_status(self) -> Dict:
        return self.colony.status()

    def get_world(self) -> Dict:
        return self.colony.world.snapshot()

    def get_best_genome(self) -> "Genome":
        alive = [c for c in self.colony.cells.values() if c.alive]
        if alive:
            return max(alive, key=lambda c: c.identity.I * c.genome.fidelity).genome
        return self._founding_genome

    def reset(self, new_genome: "Genome") -> None:
        col = self.colony
        world = col.world
        col.cells.clear()
        col.junctions.clear()
        col._junction_pairs.clear()
        col.organism = OrganismState()
        col.tick_count = 0
        col.dead_log.clear()
        world.nutrients[:] = 0.0
        world.toxins[:] = 0.0
        world.signals[:] = 0.0
        world.occupied.clear()
        world.tick_count = 0
        for sx, sy in world.sources:
            world.nutrients[sy, sx] = 80.0
        for _ in range(self._scale.prewarm_ticks):
            world.tick()
        self._founding_genome = new_genome
        col.spawn_primordial_from_genome(self._scale.primordial_cells, new_genome)

    def reseed(self, genome: "Genome") -> None:
        self.colony.spawn_primordial_from_genome(3, genome)


def _colony_worker_process(init_params: Dict, conn) -> None:
    """Entry point for each colony worker. Loops receiving commands until 'stop'."""
    worker = _ColonyWorker(init_params)
    conn.send("ready")
    while True:
        msg = conn.recv()
        tag = msg[0]
        if tag == "tick":
            global_signal = msg[1] if len(msg) > 1 else None
            conn.send(("done",) + worker.tick(global_signal=global_signal))
        elif tag == "get_emigrants":
            conn.send(("emigrants", worker.get_emigrants(msg[1])))
        elif tag == "receive_immigrant":
            worker.receive_immigrant(msg[1])
            conn.send(("immigration_done",))
        elif tag == "status":
            conn.send(("status", worker.get_status()))
        elif tag == "world":
            conn.send(("world", worker.get_world()))
        elif tag == "get_genome":
            conn.send(("genome", worker.get_best_genome()))
        elif tag == "reseed":
            worker.reseed(msg[1])
            conn.send(("reseeded",))
        elif tag == "reset":
            worker.reset(msg[1])
            conn.send(("reset_done",))
        elif tag == "stop":
            break


# ─────────────────────────────────────────────────────────────
# EVOLUTION ENGINE — tournament selection between colonies
# ─────────────────────────────────────────────────────────────

class EvolutionEngine:
    """
    Runs N_COLONIES colonies in true parallel — one worker process per colony.
    Every TOURNAMENT_INTERVAL ticks: ranks by fitness, replaces the bottom half
    with mutated genomes from the winners (multilevel tournament selection).
    """

    def __init__(self, n_colonies: int, tournament_interval: int, rng: random.Random,
                 perturbation_interval: int = 200, perturbation_strength: float = 3.0,
                 _worker_cls=None):
        if _worker_cls is None:
            _worker_cls = mp.Process
        self._worker_cls = _worker_cls
        self.n_colonies = n_colonies
        self.tournament_interval = tournament_interval
        self.rng = rng
        self.tick_count = 0
        self.evolution_history: List[Dict] = []
        self._mean_fitness_history: List[float] = []
        self._stress_hypermutation: bool = False
        self._cached_best_idx: int = 0
        self._cached_fitnesses: List[float] = [0.0] * n_colonies
        self._cached_extinct: List[bool] = [False] * n_colonies
        self._cached_alive_counts: List[int] = [0] * n_colonies
        self._cached_stages: List[str] = ["solitary"] * n_colonies
        self._cached_best_status: Dict = {}
        self._cached_best_world: Dict = {}
        self._global_signal: np.ndarray = np.zeros(N_SIGNAL_CHANNELS, dtype=np.float64)
        self._collective_pressure: float = 0.0
        self._migration_count: int = 0
        self.founding_genomes: List[Genome] = []
        self._conns: List = []
        self._procs: List = []
        self._worker_init_params: List[Dict] = []

        primordial = Genome.create(rng)
        per_colony_cells = max(4, MAX_CELLS // n_colonies)

        for _ in range(n_colonies):
            founding = primordial.mutate(rng)
            col_rng_seed = rng.randint(0, 2**31)
            init_params = {
                "founding_genome": founding,
                "col_rng_seed": col_rng_seed,
                "per_colony_cells": per_colony_cells,
                "perturbation_interval": perturbation_interval,
                "perturbation_strength": perturbation_strength,
            }
            parent_conn, child_conn = mp.Pipe(duplex=True)
            proc = _worker_cls(target=_colony_worker_process,
                               args=(init_params, child_conn), daemon=True)
            proc.start()
            if _worker_cls is not threading.Thread:
                child_conn.close()
            self.founding_genomes.append(founding)
            self._conns.append(parent_conn)
            self._procs.append(proc)
            self._worker_init_params.append(init_params)

        for conn in self._conns:
            assert conn.recv() == "ready"

        self._refresh_best_cache()

    # ── Parallel tick ──────────────────────────────────────────

    def tick(self) -> None:
        for conn in self._conns:
            conn.send(("tick", self._global_signal))
        signal_summaries: List[np.ndarray] = []
        for i, conn in enumerate(self._conns):
            msg = conn.recv()
            self._cached_fitnesses[i]    = msg[1]
            self._cached_extinct[i]      = msg[2]
            self._cached_alive_counts[i] = msg[3]
            self._cached_stages[i]       = msg[4]
            if len(msg) > 5:
                signal_summaries.append(msg[5])
        self.tick_count += 1
        self._cached_best_idx = int(np.argmax(self._cached_fitnesses))
        if signal_summaries:
            self._global_signal = np.mean(signal_summaries, axis=0)
        self._apply_collective_pressure()
        if self.tick_count % MIGRATION_INTERVAL == 0:
            self._run_migration()
        if self.tick_count % self.tournament_interval == 0:
            self._run_tournament()

    # ── Tournament ─────────────────────────────────────────────

    def _run_tournament(self) -> None:
        fitnesses = list(self._cached_fitnesses)
        ranked = sorted(range(self.n_colonies), key=lambda i: fitnesses[i], reverse=True)
        n_winners = self.n_colonies // 2
        winners = ranked[:n_winners]
        losers  = ranked[n_winners:]

        current_mean = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
        if len(self._mean_fitness_history) >= 1:
            prev_mean = self._mean_fitness_history[-1]
            if prev_mean > 1e-9 and current_mean < prev_mean * 0.80:
                self._stress_hypermutation = True
            else:
                self._stress_hypermutation = False
        self._mean_fitness_history.append(current_mean)
        if len(self._mean_fitness_history) > 10:
            self._mean_fitness_history.pop(0)

        best_idx = winners[0]

        # Request genomes from all needed winners simultaneously
        needed_winners = sorted({winners[rank % len(winners)] for rank in range(len(losers))})
        for wi in needed_winners:
            self._conns[wi].send(("get_genome",))
        winner_genomes: Dict[int, Genome] = {}
        for wi in needed_winners:
            _, genome = self._conns[wi].recv()
            winner_genomes[wi] = genome

        best_genome_obj = winner_genomes.get(best_idx)
        best_genome_snap: Optional[Dict] = None
        if best_genome_obj is not None:
            best_genome_snap = {
                "fidelity":           round(best_genome_obj.fidelity, 3),
                "membrane_strength":  round(best_genome_obj.membrane_strength, 3),
                "metabolic_base_rate": round(best_genome_obj.metabolic_base_rate, 3),
                "neural_plasticity":  round(best_genome_obj.neural_plasticity, 4),
                "motility":           round(best_genome_obj.motility, 3),
            }

        # Get best colony status for organisms snapshot in history
        self._conns[best_idx].send(("status",))
        _, best_status = self._conns[best_idx].recv()

        # Send resets to all losers
        for rank, li in enumerate(losers):
            wi = winners[rank % len(winners)]
            donor_genome = winner_genomes.get(wi, self.founding_genomes[wi])
            new_genome = donor_genome.mutate(self.rng)
            if self._stress_hypermutation:
                new_genome.fidelity = max(0.30, new_genome.fidelity * 0.55)
            self.founding_genomes[li] = new_genome
            self._conns[li].send(("reset", new_genome))

        for li in losers:
            self._conns[li].recv()  # ("reset_done",)

        self.evolution_history.append({
            "tournament":    len(self.evolution_history) + 1,
            "tick":          self.tick_count,
            "fitnesses":     [float(round(f, 4)) for f in fitnesses],
            "winner_indices": winners,
            "loser_indices":  losers,
            "best_fitness":  float(round(fitnesses[best_idx], 4)),
            "mean_fitness":  float(round(current_mean, 4)),
            "best_genome":   best_genome_snap,
            "best_organisms": best_status.get("organisms", [])[:3],
        })
        if len(self.evolution_history) > 20:
            self.evolution_history.pop(0)

    def _apply_collective_pressure(self) -> None:
        n_ext = sum(1 for e in self._cached_extinct if e)
        mean_fit = sum(self._cached_fitnesses) / self.n_colonies
        self._collective_pressure = (n_ext / self.n_colonies) * max(0.0, 1.0 - mean_fit)
        if self._collective_pressure > 0.0:
            self._global_signal[SIGNAL_IDX["death_trace"]] = np.clip(
                self._global_signal[SIGNAL_IDX["death_trace"]] + self._collective_pressure * 50.0, 0.0, 100.0)
            self._global_signal[SIGNAL_IDX["toxin_alarm"]] = np.clip(
                self._global_signal[SIGNAL_IDX["toxin_alarm"]] + self._collective_pressure * 30.0, 0.0, 100.0)

    # ── Migration ──────────────────────────────────────────────

    def _run_migration(self) -> None:
        active = [(i, self._cached_alive_counts[i])
                  for i in range(self.n_colonies) if not self._cached_extinct[i]]
        if len(active) < 2:
            return
        active.sort(key=lambda t: t[1], reverse=True)
        mid = len(active) // 2
        sources = [idx for idx, _ in active[:mid]]
        targets  = [idx for idx, _ in active[mid:]]
        for src in sources:
            self._conns[src].send(("get_emigrants", 1))
        emigrants_by_source: Dict[int, List[Dict]] = {}
        for src in sources:
            _, emigrant_list = self._conns[src].recv()
            emigrants_by_source[src] = emigrant_list
        sent_targets: List[int] = []
        for rank, src in enumerate(sources):
            if not emigrants_by_source.get(src):
                continue
            tgt = targets[rank % len(targets)]
            self._conns[tgt].send(("receive_immigrant", emigrants_by_source[src][0]))
            sent_targets.append(tgt)
        for tgt in sent_targets:
            self._conns[tgt].recv()
            self._migration_count += 1

    # ── Cache / HTTP helpers ───────────────────────────────────

    def _refresh_best_cache(self) -> None:
        """Fetch status and world snapshot from the best worker. Main thread only."""
        best = self._cached_best_idx
        self._conns[best].send(("status",))
        _, self._cached_best_status = self._conns[best].recv()
        self._conns[best].send(("world",))
        _, self._cached_best_world = self._conns[best].recv()

    def best_colony_idx(self) -> int:
        return self._cached_best_idx

    def best_colony_status(self) -> Dict:
        if not self._cached_best_status:
            self._refresh_best_cache()
        return self._cached_best_status

    def best_world_snapshot(self) -> Dict:
        if not self._cached_best_world:
            self._refresh_best_cache()
        return self._cached_best_world

    def reseed_extinct_colonies(self) -> None:
        extinct_indices = [i for i, ext in enumerate(self._cached_extinct) if ext]
        for i in extinct_indices:
            self._conns[i].send(("reseed", self.founding_genomes[i]))
        for i in extinct_indices:
            self._conns[i].recv()  # ("reseeded",)

    def shutdown(self) -> None:
        for conn in self._conns:
            conn.send(("stop",))
        for proc in self._procs:
            proc.join(timeout=2.0)

    def status(self) -> Dict:
        return {
            "tick":                     self.tick_count,
            "n_colonies":               self.n_colonies,
            "tournament_interval":      self.tournament_interval,
            "tournaments_run":          len(self.evolution_history),
            "colony_fitnesses":         [float(round(f, 4)) for f in self._cached_fitnesses],
            "best_colony_idx":          self._cached_best_idx,
            "colony_sizes":             list(self._cached_alive_counts),
            "colony_stages":            list(self._cached_stages),
            "founding_genome_fidelities": [float(round(g.fidelity, 3)) for g in self.founding_genomes],
            "evolution_history":        self.evolution_history[-5:],
        }


# ─────────────────────────────────────────────────────────────
# HTTP SERVER / DASHBOARD
# ─────────────────────────────────────────────────────────────

EVOLUTION_ENGINE: Optional["EvolutionEngine"] = None

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>nx-1 1.0 — LDNC</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #060a0f; color: #c8d8e8; font-family: 'Courier New', monospace; font-size: 12px; }
  #header { padding: 12px 20px; background: #0a1520; border-bottom: 1px solid #1a3a5a; display: flex; align-items: center; gap: 20px; }
  #header h1 { font-size: 16px; color: #40c8ff; letter-spacing: 2px; }
  #header .subtitle { color: #4a7090; font-size: 10px; }
  .pulse { width: 8px; height: 8px; background: #40c8ff; border-radius: 50%; animation: pulse 1.5s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.7)} }
  #main { display: grid; grid-template-columns: 280px 1fr 280px; height: calc(100vh - 45px); }
  .panel { overflow-y: auto; padding: 10px; border-right: 1px solid #0f2030; }
  .panel:last-child { border-right: none; border-left: 1px solid #0f2030; }
  #world-panel { display: flex; flex-direction: column; align-items: center; padding: 10px; }
  .section-title { color: #2a8abf; font-size: 10px; letter-spacing: 2px; text-transform: uppercase; padding: 6px 0 4px; border-bottom: 1px solid #0f2030; margin-bottom: 6px; }
  .stat-row { display: flex; justify-content: space-between; padding: 2px 0; }
  .stat-label { color: #4a7090; }
  .stat-value { color: #80d0ff; }
  .bar-wrap { margin: 2px 0; }
  .bar-label { color: #4a7090; font-size: 10px; margin-bottom: 1px; display: flex; justify-content: space-between; }
  .bar-bg { background: #0a1520; height: 6px; border-radius: 3px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
  .bar-atp { background: linear-gradient(90deg, #1a5a8a, #40c8ff); }
  .bar-mem { background: linear-gradient(90deg, #2a1a5a, #8040ff); }
  .bar-boundary { background: linear-gradient(90deg, #1a4a2a, #40c080); }
  .bar-stress { background: linear-gradient(90deg, #4a2a1a, #ff8040); }
  .bar-identity { background: linear-gradient(90deg, #3a1a3a, #ff40ff); }
  .bar-damage { background: linear-gradient(90deg, #4a1a1a, #ff4040); }
  .bar-neural { background: linear-gradient(90deg, #1a3a4a, #40ffff); }
  .cell-card { background: #0a1520; border: 1px solid #0f2030; border-radius: 4px; padding: 8px; margin-bottom: 6px; cursor: pointer; transition: border-color 0.2s; }
  .cell-card:hover, .cell-card.selected { border-color: #40c8ff; }
  .cell-id { color: #40c8ff; font-size: 11px; }
  .cell-phase { font-size: 9px; padding: 1px 5px; border-radius: 3px; display: inline-block; margin-left: 5px; }
  .phase-active { background: #1a3a1a; color: #40c040; }
  .phase-developing { background: #1a2a3a; color: #4080ff; }
  .phase-replicating { background: #3a2a1a; color: #ffa040; }
  .phase-stressed { background: #3a1a1a; color: #ff6060; }
  .phase-repairing { background: #2a2a1a; color: #e0c040; }
  .phase-aging { background: #2a1a2a; color: #a060a0; }
  .phase-dying { background: #3a0a0a; color: #ff2020; }
  .blocks-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; margin-top: 6px; }
  .block-mini { background: #060f18; border: 1px solid #0f2030; border-radius: 3px; padding: 4px; }
  .block-mini-title { color: #2a6a9a; font-size: 9px; margin-bottom: 2px; }
  canvas { border: 1px solid #0f2030; border-radius: 4px; }
  #colony-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; margin-bottom: 10px; }
  .colony-stat { background: #0a1520; border: 1px solid #0f2030; border-radius: 3px; padding: 6px; text-align: center; }
  .colony-stat .val { color: #40c8ff; font-size: 18px; }
  .colony-stat .lbl { color: #4a7090; font-size: 9px; }
  .event { color: #4a6a4a; font-size: 10px; padding: 1px 0; border-bottom: 1px solid #060a0f; }
  .event.repro { color: #6a4a6a; }
  .event.death { color: #6a2a2a; }
  .priorities-bar { display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin: 4px 0; }
  .pri-maint { background: #40c8ff; }
  .pri-repair { background: #40c080; }
  .pri-action { background: #ffa040; }
  .pri-repro  { background: #ff40ff; }
  #tick-counter { color: #4a7090; font-size: 11px; }
  .death-entry { background: #150505; border: 1px solid #3a0a0a; border-radius: 3px; padding: 4px; margin: 2px 0; font-size: 10px; }
  .death-cause { color: #ff6060; }
  .gen-badge { background: #0f1a2a; color: #4080c0; font-size: 9px; padding: 1px 4px; border-radius: 2px; }
  .closure-ring { display: flex; gap: 3px; margin: 4px 0; flex-wrap: wrap; }
  .closure-node { font-size: 9px; padding: 1px 4px; border-radius: 10px; }
  #speed-control { color: #4a7090; margin-left: auto; }
  button { background: #0a2030; border: 1px solid #1a4060; color: #60a0c0; padding: 3px 8px; border-radius: 3px; cursor: pointer; font-size: 10px; }
  button:hover { background: #0f2a40; }
  .bar-fitness { background: linear-gradient(90deg, #1a3a1a, #40ff80); }
  .bar-best    { background: linear-gradient(90deg, #2a2a1a, #ffdd00); }
</style>
</head>
<body>
<div id="header">
  <div class="pulse"></div>
  <h1>nx-1</h1>
  <div class="subtitle">LDNC — 9 blocks | B1-B9 | M1-M3 | H1-H4 | D1-D7</div>
  <span id="tick-counter">tick 0</span>
  <div id="speed-control">
    speed: <button onclick="setSpeed(50)">1×</button>
           <button onclick="setSpeed(20)">2×</button>
           <button onclick="setSpeed(5)">5×</button>
  </div>
</div>
<div id="main">
  <!-- Left panel: Colony -->
  <div class="panel" id="left-panel">
    <div class="section-title">Colony</div>
    <div id="colony-stats"></div>
    <div class="section-title">Evolution — Tournament</div>
    <div id="evo-stats"></div>
    <div id="evo-fitness-bars"></div>
    <div class="section-title">Living cells</div>
    <div id="cells-list"></div>
    <div class="section-title">Recent deaths</div>
    <div id="deaths-log"></div>
  </div>

  <!-- Center panel: World + chart -->
  <div id="world-panel">
    <canvas id="world-canvas" width="400" height="400"></canvas>
    <div style="width:400px; margin-top:8px;">
      <div class="section-title">Identity I — colony</div>
      <canvas id="identity-chart" width="400" height="80"></canvas>
    </div>
  </div>

  <!-- Right panel: Selected cell -->
  <div class="panel" id="right-panel">
    <div class="section-title">Selected cell</div>
    <div id="cell-detail">
      <div style="color:#4a7090; padding: 20px 0; text-align:center;">Select a cell</div>
    </div>
  </div>
</div>

<script>
let selectedId = null;
let data = {};
let identityHistory = [];
let speed = 50;

function setSpeed(ms) { speed = ms; }

function barHtml(cls, val) {
  val = Math.max(0, Math.min(1, val || 0));
  return `<div class="bar-bg"><div class="bar-fill ${cls}" style="width:${(val*100).toFixed(1)}%"></div></div>`;
}

function barRow(label, cls, val) {
  return `<div class="bar-wrap">
    <div class="bar-label"><span>${label}</span><span>${(val*100).toFixed(0)}%</span></div>
    ${barHtml(cls, val)}
  </div>`;
}

function phaseTag(p) {
  return `<span class="cell-phase phase-${p}">${p}</span>`;
}

function renderColonyStats(d) {
  const s = d;
  const org = s.organism || {};
  document.getElementById('colony-stats').innerHTML = `
    <div class="colony-stat"><div class="val">${s.alive}</div><div class="lbl">alive</div></div>
    <div class="colony-stat"><div class="val">${(s.avg_identity_I*100).toFixed(0)}%</div><div class="lbl">mean I</div></div>
    <div class="colony-stat"><div class="val">${s.junction_count||0}</div><div class="lbl">junctions</div></div>
    <div class="colony-stat"><div class="val">${org.organism_count||0}</div><div class="lbl">organisms</div></div>
    <div class="colony-stat"><div class="val">${((org.collective_identity||0)*100).toFixed(0)}%</div><div class="lbl">body I</div></div>
    <div class="colony-stat"><div class="val">${((org.boundary_closure||0)*100).toFixed(0)}%</div><div class="lbl">body boundary</div></div>
    <div class="colony-stat"><div class="val">${s.tick}</div><div class="lbl">tick</div></div>
  `;
  document.getElementById('tick-counter').textContent = `tick ${s.tick}`;
}

function renderCells(cells) {
  const list = document.getElementById('cells-list');
  list.innerHTML = cells.map(c => {
    const S = c.S || {};
    const sel = c.id === selectedId ? 'selected' : '';
    return `<div class="cell-card ${sel}" onclick="selectCell('${c.id}')">
      <div><span class="cell-id">#${c.id}</span>${phaseTag(c.phase)}
        <span class="gen-badge">G${c.generation}</span>
        <span style="color:#4a7090;float:right">t=${c.age}</span>
      </div>
      <div style="color:#4a7090;font-size:10px">type: ${c.cell_type || 'n/a'} | org: ${c.multicellular?.organism_id || '-'}</div>
      ${barRow('I identity', 'bar-identity', S.I)}
      ${barRow('ATP', 'bar-atp', S.A)}
      ${barRow('Boundary', 'bar-boundary', S.C)}
    </div>`;
  }).join('');
}

function renderDeaths(deaths) {
  document.getElementById('deaths-log').innerHTML = deaths.slice(-5).reverse().map(d => `
    <div class="death-entry">
      <span class="death-cause">${d.cause}</span>
      <span style="color:#4a7090"> t=${d.age} G${d.gen}</span>
    </div>
  `).join('');
}

function renderCellDetail(c) {
  if (!c) return;
  const S = c.S || {};
  const B3 = c.B3_homeostasis || {};
  const B4 = c.B4_neural || {};
  const B5 = c.B5_memory || {};
  const B9 = c.B9_identity || {};
  const B7 = c.B7_reproduction || {};
  const B8 = c.B8_communication || {};
  const MC = c.multicellular || {};
  const pri = B3.priorities || {};

  const closureNodes = [
    {lbl:'M→A', val: S.A, tip:'metabolism→ATP'},
    {lbl:'A→R', val: 1-(S.X||0), tip:'ATP→repair'},
    {lbl:'R→B1', val: S.C, tip:'rep→boundary'},
    {lbl:'B1→M', val: S.C*(S.C||0), tip:'boundary→metabolism'},
    {lbl:'Cog→', val: B4.excitation||0, tip:'cogn→action'}
  ];
  const closureHtml = closureNodes.map(n => {
    const v = Math.max(0,Math.min(1,n.val||0));
    const r = Math.floor(40 + (1-v)*160); const g = Math.floor(40 + v*160); const b = 80;
    return `<div class="closure-node" style="background:rgb(${r},${g},${b}20);border:1px solid rgb(${r},${g},${b}60);color:rgb(${r},${g},${b})"
      title="${n.tip}">${n.lbl} ${(v*100).toFixed(0)}%</div>`;
  }).join('');

  document.getElementById('cell-detail').innerHTML = `
    <div style="margin-bottom:6px">
      <span class="cell-id">#${c.id}</span>${phaseTag(c.phase)}
      <span class="gen-badge" style="margin-left:5px">Generation ${c.generation}</span>
      <div style="color:#4a7090;font-size:10px">age ${c.age} ticks | pos (${c.position.x},${c.position.y}) | org ${MC.organism_id || '-'}</div>
    </div>

    <div class="section-title">B9 Identity — I = ${((S.I||0)*100).toFixed(0)}%</div>
    ${barRow('I total', 'bar-identity', S.I)}
    <div style="font-size:10px;color:#4a7090;margin:4px 0">Organizational closure:</div>
    <div class="closure-ring">${closureHtml}</div>

    <div class="section-title">B1 Boundary</div>
    ${barRow('integrity', 'bar-boundary', S.C)}
    ${barRow('permanent damage', 'bar-damage', c.B1_boundary?.c_permanent_damage)}

    <div class="section-title">B2 Metabolism</div>
    ${barRow('ATP', 'bar-atp', S.A)}
    ${barRow('waste W', 'bar-damage', S.W)}
    ${barRow('efficiency η', 'bar-neural', c.B2_metabolism?.eta_metabolic)}
    <div class="stat-row"><span class="stat-label">masa_struct</span><span class="stat-value">${(S.M||0).toFixed(1)}</span></div>
    <div class="stat-row"><span class="stat-label">mat_reparativo</span><span class="stat-value">${(S.P||0).toFixed(1)}</span></div>

    <div class="section-title">B3 Homeostasis (regulatory network)</div>
    <div style="font-size:10px;color:#4a7090;margin:2px 0">Priorities ← w_reg @ errors:</div>
    <div class="priorities-bar">
      <div class="pri-maint" style="width:${((pri.maintenance||0)*100).toFixed(0)}%" title="maintenance"></div>
      <div class="pri-repair" style="width:${((pri.repair||0)*100).toFixed(0)}%" title="repair"></div>
      <div class="pri-action" style="width:${((pri.action||0)*100).toFixed(0)}%" title="action"></div>
      <div class="pri-repro" style="width:${((pri.reproduction||0)*100).toFixed(0)}%" title="reproduction"></div>
    </div>
    <div style="display:flex;gap:8px;font-size:9px;margin:2px 0">
      <span style="color:#40c8ff">■ maint ${((pri.maintenance||0)*100).toFixed(0)}%</span>
      <span style="color:#40c080">■ rep ${((pri.repair||0)*100).toFixed(0)}%</span>
      <span style="color:#ffa040">■ act ${((pri.action||0)*100).toFixed(0)}%</span>
      <span style="color:#ff40ff">■ repr ${((pri.reproduction||0)*100).toFixed(0)}%</span>
    </div>
    ${barRow('systemic stress', 'bar-stress', B3.stress)}

    <div class="section-title">B4 Neural substrate</div>
    ${barRow('excitation', 'bar-neural', B4.excitation)}
    ${barRow('integrity', 'bar-boundary', B4.integrity)}
    ${barRow('prediction error', 'bar-stress', B4.prediction_error)}
    <div class="stat-row"><span class="stat-label">action_bias</span>
      <span class="stat-value">[${(B4.action_bias||[0,0,0]).map(v=>v.toFixed(2)).join(', ')}]</span></div>

    <div class="section-title">B5 Material memory</div>
    ${barRow('integrity', 'bar-mem', B5.h_integrity)}
    ${barRow('corruption rate', 'bar-damage', (B5.corruption_rate||0)*20)}
    <div class="stat-row"><span class="stat-label">h_regulatory</span>
      <span class="stat-value">[${(B5.h_regulatory||[]).map(v=>v.toFixed(2)).join(',')}]</span></div>

    <div class="section-title">B7 Reproduction</div>
    ${barRow('maturity', 'bar-atp', B7.maturity)}
    ${barRow('failure risk', 'bar-damage', B7.failure_risk)}
    <div class="stat-row"><span class="stat-label">replicating</span>
      <span class="stat-value" style="color:${B7.replicating?'#ffa040':'#4a7090'}">${B7.replicating?'YES (stage '+B7.d_stage+')':'no'}</span></div>

    <div class="section-title">B8 Biosemiotic communication</div>
    ${barRow('signal load', 'bar-neural', B8.signal_load || 0)}
    ${barRow('coherence', 'bar-mem', B8.coherence || 0)}
    <div class="stat-row"><span class="stat-label">social move</span>
      <span class="stat-value">[${(B8.social_move||[0,0]).map(v=>v.toFixed(2)).join(', ')}]</span></div>
    <div class="stat-row"><span class="stat-label">received</span>
      <span class="stat-value">${Object.entries(B8.received||{}).filter(([k,v])=>v>0.02).map(([k,v])=>k+':'+v.toFixed(2)).join(' | ') || 'silence'}</span></div>

    <div class="section-title">Multicellular layer</div>
    ${barRow('adhesion', 'bar-boundary', MC.attachment_strength || 0)}
    ${barRow('spike', 'bar-neural', MC.spike_output || 0)}
    ${barRow('cheater', 'bar-damage', MC.cheater_score || 0)}
    <div class="stat-row"><span class="stat-label">junctions</span><span class="stat-value">${MC.junction_count || 0}</span></div>
    <div class="stat-row"><span class="stat-label">support</span>
      <span class="stat-value">+${(MC.provided_support||0).toFixed(2)} / -${(MC.received_support||0).toFixed(2)}</span></div>

    <div class="section-title">Recent events</div>
    ${(c.events||[]).slice(-8).reverse().map(e => {
      const cls = e.includes('REPRO')||e.includes('OFFSPRING') ? 'repro' : e.includes('DEAD') ? 'death' : '';
      return `<div class="event ${cls}">${e}</div>`;
    }).join('')}
  `;
}

function drawWorld(worldData, cells) {
  const canvas = document.getElementById('world-canvas');
  const ctx = canvas.getContext('2d');
  const W = worldData.W, H = worldData.H;
  const pw = canvas.width / W, ph = canvas.height / H;

  const nut = worldData.nutrients;
  const tox = worldData.toxins;

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const n = Math.min(1, nut[y][x] / 100);
      const t = Math.min(1, tox[y][x] / 40);
      const r = Math.floor(t * 120);
      const g = Math.floor(n * 80 + 10);
      const b = Math.floor(n * 20 + 15);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(x * pw, y * ph, pw, ph);
    }
  }

  // Draw sources
  (worldData.sources || []).forEach(([sx, sy]) => {
    ctx.strokeStyle = 'rgba(64,200,255,0.3)';
    ctx.lineWidth = 0.5;
    ctx.strokeRect(sx*pw, sy*ph, pw, ph);
  });

  const byId = Object.fromEntries(cells.map(c => [c.id, c]));
  (data.junctions || []).forEach(j => {
    const a = byId[j.a], b = byId[j.b];
    if (!a || !b) return;
    const ax = a.position.x * pw + pw/2, ay = a.position.y * ph + ph/2;
    const bx = b.position.x * pw + pw/2, by = b.position.y * ph + ph/2;
    const colors = {adhesion:'#60c080', metabolic:'#40c8ff', gap:'#c0c040', synaptic:'#ff60c0'};
    ctx.strokeStyle = colors[j.kind] || '#808080';
    ctx.globalAlpha = Math.max(0.15, Math.min(0.8, j.strength || 0.2));
    ctx.lineWidth = j.kind === 'synaptic' ? 1.2 : 0.8;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
    ctx.globalAlpha = 1;
  });

  // Draw cells
  cells.forEach(c => {
    const cx = c.position.x * pw + pw/2;
    const cy = c.position.y * ph + ph/2;
    const I = c.S?.I || 0;
    const r2 = Math.max(2, pw * 0.55);

    // glow based on identity
    ctx.shadowColor = `rgba(${Math.floor((1-I)*255), ${Math.floor(I*200)}, 255, 0.8})`;
    ctx.shadowBlur = r2 * 2;

    // phase color
    const phaseColors = {
      active: '#40c080', developing: '#4080ff', replicating: '#ffa040',
      stressed: '#ff6040', repairing: '#e0c040', aging: '#a060a0',
      dying: '#ff2020', dead: '#202020'
    };
    ctx.fillStyle = phaseColors[c.phase] || '#ffffff';
    ctx.beginPath();
    ctx.arc(cx, cy, r2, 0, Math.PI*2);
    ctx.fill();
    ctx.shadowBlur = 0;

    // highlight if selected
    if (c.id === selectedId) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx, cy, r2 + 2, 0, Math.PI*2);
      ctx.stroke();
    }
  });
}

function drawIdentityChart(history) {
  const canvas = document.getElementById('identity-chart');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#060a0f';
  ctx.fillRect(0, 0, W, H);

  if (history.length < 2) return;

  // threshold line M3
  ctx.strokeStyle = 'rgba(255,40,40,0.3)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, H * (1 - 0.18));
  ctx.lineTo(W, H * (1 - 0.18));
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = '#ff40ff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  history.forEach((v, i) => {
    const x = (i / (history.length - 1)) * W;
    const y = H * (1 - Math.max(0, Math.min(1, v)));
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function selectCell(id) {
  selectedId = id;
  const c = (data.cells || []).find(c => c.id === id);
  renderCellDetail(c);
}

function renderEvolution(evo) {
  if (!evo || evo.loading || !evo.n_colonies) {
    document.getElementById('evo-stats').innerHTML = '<div style="color:#4a7090;padding:4px 0">initializing…</div>';
    document.getElementById('evo-fitness-bars').innerHTML = '';
    return;
  }
  const fitnesses = evo.colony_fitnesses || [];
  const best_idx  = evo.best_colony_idx || 0;
  const maxFit    = Math.max(...fitnesses, 0.001);
  const meanFit   = fitnesses.reduce((a, b) => a + b, 0) / Math.max(1, fitnesses.length);
  document.getElementById('evo-stats').innerHTML = `
    <div class="stat-row"><span class="stat-label">tournaments</span><span class="stat-value">${evo.tournaments_run}</span></div>
    <div class="stat-row"><span class="stat-label">colonies</span><span class="stat-value">${evo.n_colonies}</span></div>
    <div class="stat-row"><span class="stat-label">best fit</span><span class="stat-value">${(fitnesses[best_idx]||0).toFixed(3)}</span></div>
    <div class="stat-row"><span class="stat-label">mean fit</span><span class="stat-value">${meanFit.toFixed(3)}</span></div>
  `;
  document.getElementById('evo-fitness-bars').innerHTML = fitnesses.map((f, i) => {
    const pct   = (f / maxFit * 100).toFixed(1);
    const cls   = i === best_idx ? 'bar-best' : 'bar-fitness';
    const stage = (evo.colony_stages || [])[i] || '?';
    const size  = (evo.colony_sizes  || [])[i] || 0;
    return `<div class="bar-wrap">
      <div class="bar-label"><span>C${i} ${stage.substring(0,3)} (${size})</span><span>${f.toFixed(2)}</span></div>
      <div class="bar-bg"><div class="bar-fill ${cls}" style="width:${pct}%"></div></div>
    </div>`;
  }).join('');
}

async function fetchEvolution() {
  try {
    const res = await fetch('/evolution');
    renderEvolution(await res.json());
  } catch(e) {}
  setTimeout(fetchEvolution, 2000);
}

async function fetchAndRender() {
  try {
    const [statusRes, worldRes] = await Promise.all([
      fetch('/status'), fetch('/world')
    ]);
    data = await statusRes.json();
    const worldData = await worldRes.json();

    renderColonyStats(data);
    renderCells(data.cells || []);
    renderDeaths(data.recent_deaths || []);
    drawWorld(worldData, data.cells || []);

    // identity history
    identityHistory.push(data.avg_identity_I || 0);
    if (identityHistory.length > 200) identityHistory.shift();
    drawIdentityChart(identityHistory);

    // update selected cell if still alive
    if (selectedId) {
      const c = (data.cells || []).find(c => c.id === selectedId);
      if (c) renderCellDetail(c);
      else selectedId = null;
    }
  } catch(e) {}
  setTimeout(fetchAndRender, speed);
}

fetchAndRender();
fetchEvolution();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', APPLICATION_JSON)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            if EVOLUTION_ENGINE:
                body = json.dumps(EVOLUTION_ENGINE.best_colony_status())
            else:
                body = '{"tick":0,"alive":0,"cells":[],"loading":true}'
            self.wfile.write(body.encode())
        elif self.path == '/world':
            self.send_response(200)
            self.send_header('Content-Type', APPLICATION_JSON)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            if EVOLUTION_ENGINE:
                body = json.dumps(EVOLUTION_ENGINE.best_world_snapshot())
            else:
                body = '{"cells":[],"loading":true}'
            self.wfile.write(body.encode())
        elif self.path == '/evolution':
            self.send_response(200)
            self.send_header('Content-Type', APPLICATION_JSON)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            if EVOLUTION_ENGINE:
                body = json.dumps(EVOLUTION_ENGINE.status())
            else:
                body = '{"tick":0,"n_colonies":0,"tournaments_run":0,"loading":true}'
            self.wfile.write(body.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        # Silence per-request HTTP logs; dashboard status is reported by the simulation loop.
        pass


def run_server(port: int = 8765):
    import socket as _socket
    class _Server(HTTPServer):
        allow_reuse_address = True
        def server_bind(self):
            self.socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            if hasattr(_socket, 'SO_REUSEPORT'):
                self.socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEPORT, 1)
            super().server_bind()
    server = _Server(('', port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server

def main():
    global EVOLUTION_ENGINE, MAX_CELLS, N_COLONIES, TOURNAMENT_INTERVAL, NP_RNG

    ap = argparse.ArgumentParser(description="nx-1 1.0 — LDNC")
    ap.add_argument("--max-cells",          type=int,   default=MAX_CELLS,
                    help=f"Total cell budget across all colonies (default {MAX_CELLS})")
    ap.add_argument("--n-colonies",         type=int,   default=N_COLONIES,
                    help=f"Number of parallel colonies (default {N_COLONIES})")
    ap.add_argument("--tournament-interval",type=int,   default=TOURNAMENT_INTERVAL,
                    help=f"Ticks between selection events (default {TOURNAMENT_INTERVAL})")
    ap.add_argument("--port",               type=int,   default=8765,
                    help="Dashboard HTTP port (default 8765)")
    ap.add_argument("--seed",               type=int,   default=42,
                    help="RNG seed (default 42)")
    ap.add_argument("--perturbation-interval", type=int, default=200,
                    help="Ticks between environmental perturbations (0=off, default 200)")
    ap.add_argument("--perturbation-strength", type=float, default=3.0,
                    help="Toxin pulse strength on perturbation (default 3.0)")
    ap.add_argument("--speed", type=str,
                    choices=list(SPEED_INTERVALS.keys()),
                    default="real-time",
                    help="Simulation speed relative to wall clock (default: real-time = 1 tick/s)")
    args = ap.parse_args()

    MAX_CELLS           = args.max_cells
    N_COLONIES          = args.n_colonies
    TOURNAMENT_INTERVAL = args.tournament_interval

    print("=" * 60)
    print("  nx-1 1.0 — LDNC + EvolutionEngine")
    print("  B1-B9 | M1-M3 | H1-H4 | D1-D7 | Multilevel tournament")
    print(f"  {N_COLONIES} colonies | tournament every {TOURNAMENT_INTERVAL} ticks | max_cells={MAX_CELLS:,}")
    print("=" * 60)

    rng = random.Random(args.seed)
    NP_RNG = np.random.default_rng(args.seed)

    server = run_server(args.port)
    print(f"\n  Dashboard  → http://localhost:{args.port}")
    print(f"  Status     → http://localhost:{args.port}/status")
    print(f"  World      → http://localhost:{args.port}/world")
    print(f"  Evolution  → http://localhost:{args.port}/evolution")
    print(f"\n  Initializing {N_COLONIES} colonies...")

    EVOLUTION_ENGINE = EvolutionEngine(
        n_colonies=N_COLONIES,
        tournament_interval=TOURNAMENT_INTERVAL,
        rng=rng,
        perturbation_interval=args.perturbation_interval,
        perturbation_strength=args.perturbation_strength,
    )

    per_colony_cells = max(4, MAX_CELLS // N_COLONIES)
    scale = derive_simulation_scale(per_colony_cells)
    print(f"  Scale per colony → max_cells={scale.max_cells:,} | "
          f"world={scale.world_width}×{scale.world_height} | "
          f"sources={scale.n_sources}")
    print("\n  Starting simulation...\n")

    run_simulation_loop(server, tick_interval=speed_to_interval(args.speed))


def run_simulation_loop(server, tick_interval: float = 1.0):
    tick = 0
    window_start = time.monotonic()
    try:
        while True:
            t_start = time.monotonic()
            EVOLUTION_ENGINE.tick()
            tick += 1
            if tick % 100 == 0:
                window_elapsed = time.monotonic() - window_start
                tps = 100.0 / max(window_elapsed, 1e-9)
                EVOLUTION_ENGINE._refresh_best_cache()
                print_periodic_status(tick, tps)
                reseed_extinct_colonies()
                window_start = time.monotonic()
            elapsed = time.monotonic() - t_start
            time.sleep(max(0.0, tick_interval - elapsed))
    except KeyboardInterrupt:
        print_shutdown_summary(tick)
        EVOLUTION_ENGINE.shutdown()
        server.shutdown()


_STAGE_ABBR = {
    "solitary": "sol", "aggregate": "agg", "proto_tissue": "pro",
    "integrated": "int", "integrated_body": "bod", "extinct": "ext",
}


def print_periodic_status(tick: int, tps: float = 0.0):
    fitnesses = EVOLUTION_ENGINE._cached_fitnesses
    alive_counts = EVOLUTION_ENGINE._cached_alive_counts
    stages = EVOLUTION_ENGINE._cached_stages
    best_idx = EVOLUTION_ENGINE._cached_best_idx
    status = EVOLUTION_ENGINE._cached_best_status
    alive = status.get("alive", 0)
    avg_identity = status.get("avg_identity_I", 0.0)
    gens = {int(g) for g in status.get("generations", {}).keys()}
    stage = status.get("organism", {}).get("development_stage", "?")
    n_tournaments = len(EVOLUTION_ENGINE.evolution_history)
    n_active = sum(1 for a in alive_counts if a > 0)
    n_cols = len(fitnesses)
    migration_count = getattr(EVOLUTION_ENGINE, '_migration_count', 0)
    collective_pressure = getattr(EVOLUTION_ENGINE, '_collective_pressure', 0.0)
    print(
        f"t={time.strftime('%Y-%m-%d %H:%M:%S')} | tick={tick:6d} | tps={tps:.1f} | "
        f"active={n_active}/{n_cols} | best=col{best_idx} "
        f"fit={fitnesses[best_idx]:.3f} "
        f"mean_fit={sum(fitnesses)/n_cols:.3f} "
        f"| alive={alive:3d} I={avg_identity:.2f} "
        f"gen={max(gens) if gens else 0} "
        f"stage={stage} "
        f"| tournaments={n_tournaments} | migrations={migration_count} pressure={collective_pressure:.3f}"
    )
    fit_str  = " ".join(
        f"{'★' if i == best_idx else ' '}{fitnesses[i]:5.2f}" for i in range(n_cols)
    )
    alive_str = " ".join(
        f"{'★' if i == best_idx else ' '}{alive_counts[i]:4d}" for i in range(n_cols)
    )
    stage_str = " ".join(
        f"{'★' if i == best_idx else ' '}{_STAGE_ABBR.get(stages[i], '???')}" for i in range(n_cols)
    )
    print(f"  fit  [{fit_str}]")
    print(f"  alive[{alive_str}]")
    print(f"  stage[{stage_str}]")


def reseed_extinct_colonies():
    EVOLUTION_ENGINE.reseed_extinct_colonies()


def print_shutdown_summary(tick: int):
    n_t = len(EVOLUTION_ENGINE.evolution_history)
    print(f"\n\n  Simulation stopped. Ticks={tick} | Tournaments={n_t}")
    if EVOLUTION_ENGINE.evolution_history:
        last = EVOLUTION_ENGINE.evolution_history[-1]
        print(f"  Last tournament: best_fitness={last['best_fitness']:.3f} "
              f"mean={last['mean_fitness']:.3f}")


if __name__ == '__main__':
    main()
