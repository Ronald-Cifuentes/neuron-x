#!/usr/bin/env python3
"""
VIDA DIGITAL v6
CNDV — Célula Neuronal Digital Viva

Implementación estricta del documento arquitectónico formal.

B1: Frontera activa          — estado completo, permeabilidad dinámica
B2: Metabolismo              — cadena 5 etapas, sin conversión perfecta
B3: Homeostasis              — red regulatoria con señales de error (sin ifs hardcoded)
B4: Sustrato cognitivo       — red recurrente + STDP + degradación
B5: Memoria material         — 4 subtipos causales: estructural/regulatoria/adaptiva/heredable
B6: Reparación               — repara TODOS los bloques, compite con reproducción
B7: Reproducción             — 4 tipos herencia, fallo posible, desarrollo como proceso
B8: Acoplamiento ecológico   — mundo espacial, acción guiada por neuronal
B9: Identidad                — variable I computada, muerte M3 organizacional

S(t) = {R_ext, R_int, A, M, P, W, X, G, T, C, I}
Muerte: M1 metabólica | M2 estructural | M3 organizacional
Degradación: D1-D7
Herencia: H1-H4
"""

import argparse, math, random, time, json, threading, uuid, copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

# ─────────────────────────────────────────────────────────────
# ESCALA DE SIMULACIÓN
# ─────────────────────────────────────────────────────────────
#
# Cambia MAX_CELLS y el resto de constantes ecológicas necesarias se
# derivan de ahí. La regla central es innegociable: si hay una ocupación
# máxima de una célula por coordenada, el mundo debe tener al menos
# max_cells coordenadas habitables.
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
EXCIT_THRESHOLD = 0.55       # spike threshold for discrete firing
REFRACTORY_PERIOD = 4        # ticks of silence after a spike


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
    Deriva las variables espaciales y tróficas que dependen de max_cells.

    Esta simulación usa un mundo denso con una célula por coordenada. Por eso
    el tamaño mínimo correcto del mundo es ceil(sqrt(max_cells))². Para valores
    enormes, la arquitectura densa actual deja de ser ejecutable en memoria; en
    ese caso fallamos temprano en vez de mentir con una colonia imposible.
    """
    max_cells = int(max_cells)
    if max_cells < 1:
        raise ValueError("MAX_CELLS debe ser >= 1")

    side = max(MIN_WORLD_SIDE, math.ceil(math.sqrt(max_cells)))
    capacity = side * side
    if capacity < max_cells:
        raise AssertionError("La capacidad espacial derivada no cubre MAX_CELLS")
    if capacity > MAX_DENSE_WORLD_CELLS:
        raise ValueError(
            "MAX_CELLS exige un mundo denso de "
            f"{capacity:,} posiciones. La arquitectura actual de matrices densas "
            "no puede garantizar supervivencia a esa escala sin cambiar el modelo espacial."
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
# CANALES BIOSEMIÓTICOS DE COMUNICACIÓN INTERCELULAR
# ─────────────────────────────────────────────────────────────
#
# No son "mensajes humanos" ni embeddings ML. Son campos causales, costosos,
# difusivos y degradables: equivalentes digitales a señales químicas locales.
# Cada canal está anclado a una variable vital de la célula, para que el
# significado no sea arbitrario sino metabólico/homeostático.
SIGNAL_CHANNELS = (
    "nutrient_beacon",      # hay recurso aprovechable aquí / camino hacia recurso
    "toxin_alarm",         # peligro externo o carga tóxica local
    "energy_need",         # déficit energético interno
    "repair_need",         # daño/frontera/memoria requieren reparación
    "reproduction_ready",  # célula madura/estable, señal ecológica de linaje
    "crowding",            # demasiada densidad local, evitar saturación espacial
    "death_trace",         # necroseñal: muerte/colapso organizacional cercano
)
SIGNAL_IDX = {name: i for i, name in enumerate(SIGNAL_CHANNELS)}
N_SIGNAL_CHANNELS = len(SIGNAL_CHANNELS)

# ─────────────────────────────────────────────────────────────
# MUNDO ESPACIAL (sustrato de B8)
# ─────────────────────────────────────────────────────────────

class SpatialWorld:
    """Mundo 2D con difusión discreta, fuentes de nutrientes y acumulación de toxinas."""

    def __init__(self, width: int = 40, height: int = 40,
                 n_sources: int = 6, rng: random.Random = None,
                 source_strength: float = 3.5):
        self.W = width
        self.H = height
        self.rng = rng or random.Random()
        self.source_strength = source_strength
        # grids
        self.nutrients = np.zeros((H, W) if False else (height, width), dtype=np.float64)
        self.toxins    = np.zeros((height, width), dtype=np.float64)
        # Campos de comunicación intercelular: C × H × W.
        # Vectorizados para mantener costo bajo: O(canales × mundo), no O(células²).
        self.signals   = np.zeros((N_SIGNAL_CHANNELS, height, width), dtype=np.float64)
        # Gradientes morfogenéticos: campos posicionales fijos para diferenciar
        # el espacio y permitir la evolución de planes corporales heredables.
        # morphogen_a: eje anterior-posterior (x=0 → 1.0, x=W-1 → 0.0)
        # morphogen_b: eje dorsal-ventral    (y=0 → 1.0, y=H-1 → 0.0)
        self.morphogen_a = np.zeros((height, width), dtype=np.float64)
        self.morphogen_b = np.zeros((height, width), dtype=np.float64)
        self.morphogen_a[:, :] = np.linspace(1.0, 0.0, width)[np.newaxis, :]
        self.morphogen_b[:, :] = np.linspace(1.0, 0.0, height)[:, np.newaxis]
        self.occupied  = {}  # (x,y) -> cell_id
        # fuentes fijas de nutrientes
        self.sources = [(self.rng.randint(2, width-3), self.rng.randint(2, height-3))
                        for _ in range(n_sources)]
        # seed inicial
        for sx, sy in self.sources:
            self.nutrients[sy, sx] = 80.0
        self.tick_count = 0

    def tick(self):
        """Difusión discreta + emisión de fuentes + evaporación."""
        # difusión nutrients
        lap = (
            np.roll(self.nutrients, 1, 0) + np.roll(self.nutrients, -1, 0) +
            np.roll(self.nutrients, 1, 1) + np.roll(self.nutrients, -1, 1) -
            4 * self.nutrients
        )
        self.nutrients += 0.08 * lap
        # difusión toxinas
        lap_t = (
            np.roll(self.toxins, 1, 0) + np.roll(self.toxins, -1, 0) +
            np.roll(self.toxins, 1, 1) + np.roll(self.toxins, -1, 1) -
            4 * self.toxins
        )
        self.toxins += 0.05 * lap_t

        # difusión de señales biosemióticas.
        # Señales = rápidas, locales, degradables. La evaporación impide memoria infinita
        # y fuerza comunicación situada en el presente ecológico.
        for ch in range(N_SIGNAL_CHANNELS):
            grid = self.signals[ch]
            lap_s = (
                np.roll(grid, 1, 0) + np.roll(grid, -1, 0) +
                np.roll(grid, 1, 1) + np.roll(grid, -1, 1) -
                4 * grid
            )
            # alarmas y muerte viajan un poco más rápido que señales cooperativas
            diff = 0.14 if ch in (SIGNAL_IDX["toxin_alarm"], SIGNAL_IDX["death_trace"]) else 0.09
            decay = 0.925 if ch in (SIGNAL_IDX["energy_need"], SIGNAL_IDX["repair_need"]) else 0.945
            self.signals[ch] += diff * lap_s
            self.signals[ch] *= decay

        # emisión fuentes
        for sx, sy in self.sources:
            self.nutrients[sy, sx] = min(120.0, self.nutrients[sy, sx] + self.source_strength)
        # evaporación suave
        self.nutrients *= 0.995
        self.toxins    *= 0.990
        np.clip(self.nutrients, 0, 120, out=self.nutrients)
        np.clip(self.toxins,    0,  60, out=self.toxins)
        np.clip(self.signals,   0, 100, out=self.signals)

        # Difusión morfogenética (lenta, α=0.04) + reimposición de fuentes de borde.
        # Los bordes actúan como fuentes/sumideros fijos → gradiente estable pero
        # no infinitamente rígido; las células pueden perturbar localmente.
        for arr in (self.morphogen_a, self.morphogen_b):
            lap = (np.roll(arr, 1, 0) + np.roll(arr, -1, 0) +
                   np.roll(arr, 1, 1) + np.roll(arr, -1, 1) - 4 * arr)
            arr += 0.04 * lap
        self.morphogen_a[:, 0]  = 1.0   # fuente anterior
        self.morphogen_a[:, -1] = 0.0   # sumidero posterior
        self.morphogen_b[0, :]  = 1.0   # fuente dorsal
        self.morphogen_b[-1, :] = 0.0   # sumidero ventral
        np.clip(self.morphogen_a, 0.0, 1.0, out=self.morphogen_a)
        np.clip(self.morphogen_b, 0.0, 1.0, out=self.morphogen_b)

        self.tick_count += 1

    def sample(self, x: int, y: int) -> Tuple[float, float]:
        x = x % self.W; y = y % self.H
        return float(self.nutrients[y, x]), float(self.toxins[y, x])

    def sample_signals(self, x: int, y: int) -> np.ndarray:
        """Vector local de señales intercelulares en (x,y)."""
        x = x % self.W; y = y % self.H
        return self.signals[:, y, x].copy()

    def sample_morphogens(self, x: int, y: int) -> Tuple[float, float]:
        """Devuelve (morphogen_a, morphogen_b) en (x,y) — información posicional."""
        x = x % self.W; y = y % self.H
        return float(self.morphogen_a[y, x]), float(self.morphogen_b[y, x])

    def deposit_signal(self, x: int, y: int, channel, amount: float):
        """Deposita una señal biosemiótica local. Canal puede ser str o índice."""
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
        """Gradiente centrado en (x,y)."""
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
        Vecindario celular local sobre el toro espacial.

        Devuelve contexto biológico suficiente para reglas de adhesión,
        diferenciación y vigilancia sin introducir búsqueda O(células²).
        """
        out: List[Dict] = []
        x = x % self.W; y = y % self.H
        radius = max(1, int(radius))
        seen: Set[str] = set()

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                if dx * dx + dy * dy > radius * radius:
                    continue
                nx = (x + dx) % self.W
                ny = (y + dy) % self.H
                cid = self.occupied.get((nx, ny))
                if cid is None or cid in seen:
                    continue
                cell = cells.get(cid)
                if cell is None or not cell.alive:
                    continue
                seen.add(cid)
                out.append({
                    "id": cid,
                    "x": nx,
                    "y": ny,
                    "dx": dx,
                    "dy": dy,
                    "distance": math.sqrt(dx * dx + dy * dy),
                    "phase": cell.phase.value,
                    "atp": cell.metabolism.atp_fraction,
                    "damage": cell.damage_X,
                    "identity": cell.identity.I,
                    "generation": cell._generation,
                    "cell_type": getattr(cell, "cell_type", None).value
                                 if hasattr(getattr(cell, "cell_type", None), "value")
                                 else str(getattr(cell, "cell_type", "unknown")),
                })
        return out

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
# GENOMA — codifica H1, H2, H3, H4
# ─────────────────────────────────────────────────────────────

@dataclass
class Genome:
    """
    Codifica los cuatro tipos de herencia del documento formal.

    H1: Herencia estructural   — propiedades de frontera y metabolismo
    H2: Herencia regulatoria   — matriz W_reg de homeostasis
    H3: Herencia cognitiva     — predisposiciones de pesos neuronales
    H4: Herencia de desarrollo — tiempos de maduración y costos
    """
    # H1: Herencia estructural
    membrane_strength:      float = 0.85
    transport_capacity:     float = 0.70
    metabolic_base_rate:    float = 0.60
    conversion_efficiency:  float = 0.55
    repair_capacity_base:   float = 0.50
    waste_tolerance:        float = 0.40
    structural_mass_cap:    float = 100.0
    repair_material_cap:    float = 60.0
    reproductive_mass_cap:  float = 80.0

    # H2: Herencia regulatoria — W_reg es una matriz 4×5
    # (4 prioridades × 5 señales de error)
    # Se almacena como lista plana de 20 floats
    W_reg_flat: List[float] = field(default_factory=lambda: [
        # fila 0: maintenance   [e_energy, e_damage, e_waste, e_boundary, e_memory]
         1.5,  0.5,  0.3,  1.2,  0.4,
        # fila 1: repair
         0.3,  1.8,  0.5,  0.8,  0.6,
        # fila 2: action
         0.8,  0.2,  0.3,  0.2,  0.5,
        # fila 3: reproduction
         0.4,  0.1,  0.2,  0.2,  0.1,
    ])

    # H3: Herencia cognitiva — sesgos iniciales de pesos neuronales
    # Red: 6 inputs → 4 hidden → 3 outputs
    # Se almacena como predisposiciones (escala de inicialización)
    neural_W_ih_flat: List[float] = field(default_factory=lambda:
        [random.gauss(0, 0.3) for _ in range(6*4)])  # 6×4 = 24
    neural_W_hh_flat: List[float] = field(default_factory=lambda:
        [random.gauss(0, 0.2) for _ in range(4*4)])  # 4×4 = 16
    neural_W_ho_flat: List[float] = field(default_factory=lambda:
        [random.gauss(0, 0.3) for _ in range(4*3)])  # 4×3 = 12

    # H3: tasa de plasticidad
    neural_plasticity:      float = 0.02
    neural_excitability:    float = 0.5

    # H4: Herencia de desarrollo
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

    # Capa heredable de comunicación: emisión, recepción y traducción corporal.
    # Esto permite que linajes distintos desarrollen dialectos compatibles o
    # parcialmente incompatibles sin usar lenguaje humano ni entrenamiento ML.
    signal_emission_strength:    float = 0.55
    signal_receptor_sensitivity: float = 0.65
    signal_selectivity:          float = 0.55
    signal_cost_factor:          float = 0.025
    signal_receptor_flat: List[float] = field(default_factory=lambda:
        [random.gauss(0, 0.25) for _ in range(N_SIGNAL_CHANNELS * 4)])

    # Morfogenética heredable: curvas de respuesta a gradientes posicionales.
    # 8 floats = 2 morfógenos × 4 tipos celulares [BOUNDARY, NEURON, METABOLIC, REPAIR].
    # Índices 0-3: respuesta al morfógeno A; 4-7: respuesta al morfógeno B.
    # Positivo → sesga hacia ese tipo; negativo → lo inhibe.
    morphogen_response_flat: List[float] = field(default_factory=lambda:
        [random.gauss(0.0, 0.25) for _ in range(8)])

    def morphogen_response(self) -> np.ndarray:
        return np.array(self.morphogen_response_flat, dtype=np.float64).reshape(2, 4)

    def W_reg(self) -> np.ndarray:
        return np.array(self.W_reg_flat, dtype=np.float64).reshape(4, 5)

    def neural_W_ih(self) -> np.ndarray:
        return np.array(self.neural_W_ih_flat, dtype=np.float64).reshape(4, 6)

    def neural_W_hh(self) -> np.ndarray:
        return np.array(self.neural_W_hh_flat, dtype=np.float64).reshape(4, 4)

    def neural_W_ho(self) -> np.ndarray:
        return np.array(self.neural_W_ho_flat, dtype=np.float64).reshape(3, 4)

    def signal_receptor_W(self) -> np.ndarray:
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
        g.W_reg_flat = [rng.gauss(0.5, 0.4) for _ in range(20)]
        # H3
        g.neural_W_ih_flat = [rng.gauss(0, 0.4) for _ in range(24)]
        g.neural_W_hh_flat = [rng.gauss(0, 0.25) for _ in range(16)]
        g.neural_W_ho_flat = [rng.gauss(0, 0.35) for _ in range(12)]
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
        return g

    def mutate(self, rng: random.Random) -> "Genome":
        child = copy.deepcopy(self)
        fid = self.fidelity

        def mf(v, lo, hi, sigma=0.05):
            if rng.random() > fid:
                v += rng.gauss(0, sigma * (hi - lo))
                if rng.random() < 0.10:  # salto grande ocasional
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
        child.W_reg_flat = ml(child.W_reg_flat, 0.12)
        # H3
        child.neural_W_ih_flat    = ml(child.neural_W_ih_flat, 0.08)
        child.neural_W_hh_flat    = ml(child.neural_W_hh_flat, 0.06)
        child.neural_W_ho_flat    = ml(child.neural_W_ho_flat, 0.08)
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
        return child


# ─────────────────────────────────────────────────────────────
# B1: FRONTERA ACTIVA
# ─────────────────────────────────────────────────────────────

@dataclass
class Boundary:
    """
    B1: Frontera activa.
    Filtra entradas, regula salidas, protege interior, sostiene individuación.
    """
    c_integrity:            float = 1.0   # integridad total [0,1]
    c_permeability_resource: float = 0.7  # cuánto recurso deja pasar
    c_permeability_signal:  float = 0.8   # señales
    c_permeability_toxin:   float = 0.15  # toxinas que entran (bajo = bueno)
    c_transport_capacity:   float = 0.7   # cap. activa de transporte
    c_maintenance_cost:     float = 0.05  # ATP por tick para mantenerse
    c_permanent_damage:     float = 0.0   # daño irrecuperable
    c_neural_gate:          float = 1.0   # modulación desde B4 [0,1]

    def degrade(self, tox_external: float, tox_internal: float,
                age_ticks: int, genome: "Genome") -> float:
        """
        D1: Degradación estructural de frontera.
        Retorna daño infligido este tick.
        """
        # daño base por toxinas externas (reducido por integridad actual)
        dmg_ext = tox_external * (1.0 - self.c_integrity * 0.5) * 0.002
        # daño por toxinas internas (residuos)
        dmg_int = tox_internal * 0.0006
        # desgaste por edad
        dmg_age = age_ticks * 0.000006
        # daño total atenuado por membrane_strength genómico
        total_dmg = (dmg_ext + dmg_int + dmg_age) / (genome.membrane_strength + 0.1)

        # aplica daño
        self.c_integrity = max(0.0, self.c_integrity - total_dmg)
        # fracción permanente del daño acumulado (irrecuperable)
        self.c_permanent_damage = min(0.6, self.c_permanent_damage + total_dmg * 0.08)

        # permeabilidad se deteriora con integridad
        self.c_permeability_resource = 0.5 + 0.5 * self.c_integrity * genome.transport_capacity
        self.c_permeability_toxin    = 0.05 + 0.3 * (1.0 - self.c_integrity)

        return total_dmg

    def repair(self, atp: float, structural: float,
               repair_fraction: float, genome: "Genome") -> Tuple[float, float]:
        """
        Repara frontera usando ATP y masa estructural.
        repair_fraction ∈ [0,1]: porción del presupuesto de reparación asignado a este bloque.
        Retorna (atp_consumed, structural_consumed).
        """
        if atp <= 0 or structural <= 0:
            return 0.0, 0.0

        max_repair = genome.repair_capacity_base * repair_fraction * 0.12
        recoverable = self.c_integrity - self.c_permanent_damage
        recovery_target = min(max_repair, 1.0 - self.c_integrity)
        if recovery_target <= 0:
            return 0.0, 0.0

        # no puedes reparar más allá del daño irrecuperable
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
        """B4 modula apertura selectiva de frontera (F3 flow)."""
        self.c_neural_gate = max(0.2, min(1.0, gate_signal))
        # La modulación ajusta permeabilidad de recurso
        self.c_permeability_resource *= (0.7 + 0.3 * self.c_neural_gate)
        self.c_permeability_resource  = min(1.0, self.c_permeability_resource)

    @property
    def is_dead(self) -> bool:
        return self.c_integrity < 0.08  # M2

    def to_dict(self) -> Dict:
        return {k: round(float(v), 4) for k, v in self.__dict__.items()}


# ─────────────────────────────────────────────────────────────
# B2: METABOLISMO
# ─────────────────────────────────────────────────────────────

@dataclass
class Metabolism:
    """
    B2: Metabolismo.
    Cadena de 5 etapas: recurso_bruto → intermedio → ATP → masa_estructural → residuo.
    Sin conversión perfecta. Siempre hay disipación, residuo, ineficiencia.
    """
    r_raw:       float = 30.0   # recurso bruto interno
    r_int:       float = 0.0    # intermediario metabólico
    a_free:      float = 40.0   # ATP / energía libre
    m_struct:    float = 20.0   # masa estructural
    p_repair:    float = 10.0   # material reparativo
    q_repro:     float = 0.0    # material reproductivo
    w_waste:     float = 0.0    # residuo / carga tóxica interna
    eta_metabolic: float = 1.0  # eficiencia metabólica [0,1] (degrada con D2)
    phi_dissipation: float = 0.05  # tasa de disipación basal

    # caps
    r_raw_cap:  float = 120.0
    a_free_cap: float = 150.0
    m_struct_cap: float = 100.0
    p_repair_cap: float = 60.0
    q_repro_cap:  float = 80.0
    w_waste_cap:  float = 80.0

    # tracking para clausura organizacional
    atp_produced_last_tick: float = 0.0
    waste_produced_last_tick: float = 0.0

    def step(self, genome: "Genome", tox_internal: float,
             maintenance_priority: float) -> Dict:
        """
        Transforma recursos en 5 etapas.
        maintenance_priority ∈ [0,1] viene de B3.
        Retorna flujos del tick.
        """
        # D2: eficiencia degradada por residuos y daño interno
        self.eta_metabolic = max(0.1, 1.0 - tox_internal * 0.015 - self.w_waste * 0.008)

        # ── ETAPA 1: recurso bruto → intermedio
        # maintenance_priority modula qué fracción del metabolismo está activa,
        # pero la tasa base debe ser suficientemente alta para sostener la célula
        rate_e1 = genome.metabolic_base_rate * self.eta_metabolic * (0.5 + 0.5 * maintenance_priority)
        converted_to_int = min(self.r_raw, self.r_raw * rate_e1 * 0.8)
        self.r_raw -= converted_to_int
        self.r_int += converted_to_int * 0.88  # 12% perdido como calor
        waste_e1    = converted_to_int * 0.04

        # ── ETAPA 2: intermedio → ATP
        rate_e2 = genome.conversion_efficiency * self.eta_metabolic
        atp_from_int = min(self.r_int, self.r_int * rate_e2 * 0.90)
        self.r_int -= atp_from_int
        atp_produced = atp_from_int * 2.2  # conversión energética
        waste_e2 = atp_from_int * 0.07
        self.a_free = min(self.a_free_cap, self.a_free + atp_produced)
        self.atp_produced_last_tick = atp_produced

        # ── ETAPA 3: ATP → masa estructural
        # La prioridad de mantenimiento dirige la fracción post-producción
        struct_alloc = 0.12 * maintenance_priority
        atp_for_struct = min(self.a_free * 0.3, self.a_free * struct_alloc)
        self.a_free -= atp_for_struct
        struct_produced = atp_for_struct * 0.65
        self.m_struct = min(self.m_struct_cap, self.m_struct + struct_produced)

        # ── ETAPA 4: ATP → material reparativo
        repair_alloc = 0.08 * maintenance_priority
        atp_for_repair = min(self.a_free * 0.20, self.a_free * repair_alloc)
        self.a_free -= atp_for_repair
        repair_produced = atp_for_repair * 0.80
        self.p_repair = min(self.p_repair_cap, self.p_repair + repair_produced)

        # ── ETAPA 5: excedente → material reproductivo
        if self.a_free > self.a_free_cap * 0.45:
            atp_for_repro = self.a_free * 0.06
            self.a_free -= atp_for_repro
            repro_produced = atp_for_repro * 0.65
            self.q_repro = min(self.q_repro_cap, self.q_repro + repro_produced)
        else:
            repro_produced = 0.0

        # ── Residuos totales (siempre hay residuo — sin conversión perfecta)
        total_waste = waste_e1 + waste_e2 + self.a_free * self.phi_dissipation * 0.005
        self.w_waste = min(self.w_waste_cap, self.w_waste + total_waste)
        self.waste_produced_last_tick = total_waste

        # ── Disipación basal (costo de existir — reducida)
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
        """B6 restaura eficiencia metabólica (D2 reversal)."""
        self.eta_metabolic = min(1.0, self.eta_metabolic + repair_amount * 0.04)
        # limpia residuos internos
        waste_cleared = min(self.w_waste, repair_amount * 2.0)
        self.w_waste -= waste_cleared
        return waste_cleared

    def to_dict(self) -> Dict:
        return {k: round(float(v), 4) for k, v in self.__dict__.items()
                if not k.endswith('_cap')}


# ─────────────────────────────────────────────────────────────
# B3: HOMEOSTASIS / REGULACIÓN INTERNA
# ─────────────────────────────────────────────────────────────

class HomeostasisRegulator:
    """
    B3: Red regulatoria.

    NO usa ifs hardcoded.
    Señales de error → W_reg (heredable) → vector de prioridades vía softmax.
    Las prioridades asignan ATP a: mantenimiento / reparación / acción / reproducción.

    G = {g_energy_error, g_damage_error, g_waste_error, g_boundary_error, g_memory_error,
         g_priority_maintenance, g_priority_repair, g_priority_action, g_priority_reproduction,
         g_stress, g_regulatory_coherence}
    """

    def __init__(self, genome: "Genome"):
        # W_reg: 4 outputs × 5 inputs — heredable (H2)
        self.W_reg = genome.W_reg().copy()
        # setpoints (pueden adaptarse vía B5 memoria regulatoria)
        self.sp_energy   = 0.65   # fracción ATP deseada
        self.sp_damage   = 0.20   # umbral de daño tolerable
        self.sp_waste    = 0.30   # fracción residuo tolerable
        self.sp_boundary = 0.75   # integridad deseada
        self.sp_memory   = 0.70   # integridad de memoria deseada

        # estado G
        self.g_energy_error   = 0.0
        self.g_damage_error   = 0.0
        self.g_waste_error    = 0.0
        self.g_boundary_error = 0.0
        self.g_memory_error   = 0.0
        self.g_stress         = 0.0
        self.g_regulatory_coherence = 1.0

        # prioridades computadas (no hardcoded)
        self.p_maintenance  = 0.35
        self.p_repair       = 0.25
        self.p_action       = 0.25
        self.p_reproduction = 0.15

        # historial para coherencia
        self._priority_history = []
        self._error_history    = []

    def update(self, met: Metabolism, boundary: Boundary,
               damage: float, mem_integrity: float,
               mem_regulatory: np.ndarray) -> Dict:
        """
        Actualiza señales de error y computa prioridades via W_reg.
        mem_regulatory viene de B5 y ajusta setpoints (plasticidad regulatoria).
        """
        # ── Señales de error (positivo = problema)
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

        # ── Ajuste de setpoints desde memoria regulatoria (B5 → B3)
        if mem_regulatory is not None and len(mem_regulatory) == 5:
            # memoria regulatoria sesga levemente los setpoints
            self.sp_energy   = np.clip(0.65 + mem_regulatory[0] * 0.05, 0.3, 0.9)
            self.sp_boundary = np.clip(0.75 + mem_regulatory[3] * 0.05, 0.4, 0.95)

        # ── Prioridades via W_reg @ error_vector → softmax
        raw = self.W_reg @ e_vec
        raw = np.clip(raw, -8, 8)
        exp = np.exp(raw - raw.max())
        priorities = exp / (exp.sum() + 1e-9)

        self.p_maintenance  = float(priorities[0])
        self.p_repair       = float(priorities[1])
        self.p_action       = float(priorities[2])
        self.p_reproduction = float(priorities[3])

        # ── Estrés sistémico = RMS de errores
        self.g_stress = float(np.sqrt(np.mean(e_vec**2)))

        # ── Coherencia regulatoria: estabilidad del vector de prioridades
        self._priority_history.append(priorities.copy())
        if len(self._priority_history) > 20:
            self._priority_history.pop(0)
        if len(self._priority_history) >= 5:
            arr = np.array(self._priority_history[-5:])
            # coherencia = 1 - varianza media de prioridades
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

    def adapt_W_reg(self, learning_signal: float):
        """
        Plasticidad regulatoria: ajuste lento de W_reg por señal de memoria (B5→B3).
        Esto permite que la estrategia de regulación evolucione dentro de la vida de la célula.
        """
        e_vec = np.array([
            self.g_energy_error, self.g_damage_error, self.g_waste_error,
            self.g_boundary_error, self.g_memory_error
        ])
        p_vec = np.array([self.p_maintenance, self.p_repair,
                          self.p_action, self.p_reproduction])
        # Hebb-like: refuerza conexiones que correlacionan con bajo estrés
        if self.g_stress < 0.2:
            delta = np.outer(p_vec, e_vec) * learning_signal * 0.001
            self.W_reg += delta
            self.W_reg = np.clip(self.W_reg, -5, 5)

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
# B4: SUSTRATO COGNITIVO-NEURONAL
# ─────────────────────────────────────────────────────────────

class NeuralCore:
    """
    B4: Sustrato cognitivo-neuronal.
    Subsistema dentro de la célula, no ES la célula.

    Arquitectura: 6 inputs → 4 hidden (recurrente) → 3 outputs
    Inputs:  [atp_frac, membrane_integrity, waste_norm, nut_local, tox_local, damage_norm]
    Outputs: [move_bias_x, move_bias_y, capture_modulation]

    + Predicción interna (error predictivo t_prediction_error)
    + STDP-like: actualiza W_ih desde error predictivo
    + Degrada con D3 cuando energía baja o daño alto

    T = {t_internal_state, t_synaptic_matrix, t_excitation,
         t_prediction_error, t_adaptive_trace, t_action_bias}
    """

    def __init__(self, genome: "Genome"):
        # Pesos sinápticos (predisposiciones heredadas H3)
        self.W_ih = genome.neural_W_ih().copy()  # 4×6
        self.W_hh = genome.neural_W_hh().copy()  # 4×4
        self.W_ho = genome.neural_W_ho().copy()  # 3×4

        # Capa de predicción: predice próximo input (6 outputs)
        self.W_pred = np.random.randn(6, 4) * 0.2  # 6×4

        # Estado interno
        self.t_internal_state   = np.zeros(4)   # h_t
        self.t_excitation       = 0.0
        self.t_prediction_error = 0.0
        self.t_adaptive_trace   = np.zeros(4)   # traza de activación
        self.t_action_bias      = np.zeros(3)   # outputs

        # Para D3: degradación cognitiva
        self.t_noise_level      = 0.0
        self.t_integrity        = 1.0

        # Parámetros heredables
        self.plasticity   = genome.neural_plasticity
        self.excitability = genome.neural_excitability

        # Historial para memoria adaptativa
        self._last_input    = np.zeros(6)
        self._input_history: List[np.ndarray] = []

    def step(self, inputs: np.ndarray, adaptive_trace_bias: np.ndarray,
             atp_available: float) -> Dict:
        """
        Un tick del sustrato neuronal.

        inputs: [atp_frac, membrane_integrity, waste_norm, nut_local_norm, tox_local_norm, damage_norm]
        adaptive_trace_bias: sesgo de B5 memoria adaptativa
        atp_available: si bajo, degrada (D3)
        """
        # ── D3: degradación cognitiva por falta de energía
        if atp_available < 5.0:
            self.t_noise_level = min(1.0, self.t_noise_level + 0.02)
            self.t_integrity = max(0.1, self.t_integrity - 0.005)
        else:
            self.t_noise_level = max(0.0, self.t_noise_level - 0.005)
            self.t_integrity = min(1.0, self.t_integrity + 0.002)

        # ruido proporcional a degradación (D3)
        noise = np.random.randn(*inputs.shape) * self.t_noise_level * 0.3

        # ── Error predictivo (predicción del input anterior)
        if np.any(self._last_input != 0):
            predicted = np.tanh(self.W_pred @ self.t_internal_state)
            self.t_prediction_error = float(np.mean(np.abs(predicted - self._last_input)))
        else:
            self.t_prediction_error = 0.0

        # ── Paso recurrente
        # sesgo de memoria adaptativa (B5 → B4: sesgos previos de activación)
        bias = adaptive_trace_bias[:4] if len(adaptive_trace_bias) >= 4 else np.zeros(4)
        noisy_input = inputs + noise
        h_new = np.tanh(
            self.W_ih @ noisy_input +
            self.W_hh @ self.t_internal_state * self.excitability +
            bias * 0.35
        )
        self.t_internal_state = h_new

        # ── Outputs
        out = np.tanh(self.W_ho @ h_new)
        self.t_action_bias = out

        # ── Excitación = norma del estado
        self.t_excitation = float(np.mean(np.abs(h_new)))

        # ── Traza adaptativa (decaying average)
        self.t_adaptive_trace = 0.92 * self.t_adaptive_trace + 0.08 * h_new

        # ── STDP-like: actualiza W_ih proporcional a error predictivo
        if self.t_prediction_error > 0.01 and atp_available > 10:
            delta_W = np.outer(h_new, noisy_input) * self.plasticity * self.t_prediction_error
            self.W_ih += delta_W
            # también actualiza W_pred
            self.W_pred += np.outer(noisy_input, h_new) * self.plasticity * 0.5
            # clamp pesos
            self.W_ih   = np.clip(self.W_ih,   -3, 3)
            self.W_pred = np.clip(self.W_pred,  -3, 3)

        self._last_input = inputs.copy()
        self._input_history.append(inputs.copy())
        if len(self._input_history) > 50:
            self._input_history.pop(0)

        # ── Outputs interpretados
        move_x    = float(out[0])  # [-1, 1]
        move_y    = float(out[1])  # [-1, 1]
        capture_m = float((out[2] + 1) / 2)  # [0, 1] modulación de captura

        return {
            "move_x": move_x,
            "move_y": move_y,
            "capture_modulation": capture_m,
            "excitation": self.t_excitation,
            "prediction_error": self.t_prediction_error,
            "integrity": self.t_integrity
        }

    def repair_weights(self, repair_amount: float):
        """B6 repara sustrato neuronal: reduce ruido, restaura integridad (D3 reversal)."""
        self.t_noise_level = max(0.0, self.t_noise_level - repair_amount * 0.1)
        self.t_integrity   = min(1.0, self.t_integrity   + repair_amount * 0.03)
        # pequeña restauración de pesos degradados
        self.W_ih   = np.clip(self.W_ih,   -3, 3)
        self.W_hh   = np.clip(self.W_hh,   -3, 3)
        self.W_ho   = np.clip(self.W_ho,   -3, 3)

    def get_heritable_predispositions(self) -> Tuple[List[float], List[float], List[float]]:
        """
        H3: Extrae predisposiciones para transmitir a descendencia.
        No transfiere aprendizaje completo — solo tendencias.
        """
        # promedia con los pesos originales genómicos para no transferir todo el aprendizaje
        W_ih_heir = (self.W_ih * 0.35).flatten().tolist()
        W_hh_heir = (self.W_hh * 0.35).flatten().tolist()
        W_ho_heir = (self.W_ho * 0.35).flatten().tolist()
        return W_ih_heir, W_hh_heir, W_ho_heir

    def to_dict(self) -> Dict:
        return {
            "excitation":       round(self.t_excitation, 4),
            "prediction_error": round(self.t_prediction_error, 4),
            "integrity":        round(self.t_integrity, 4),
            "noise_level":      round(self.t_noise_level, 4),
            "action_bias":      [round(float(v), 3) for v in self.t_action_bias],
            "adaptive_trace":   [round(float(v), 3) for v in self.t_adaptive_trace]
        }


# ─────────────────────────────────────────────────────────────
# B5: MEMORIA MATERIAL
# ─────────────────────────────────────────────────────────────

class MaterialMemory:
    """
    B5: Memoria material.
    4 subtipos, todos causalmente activos, todos degradables, todos costosos.

    h_structural:  historial de integridad de frontera → alimenta prioridad de reparación
    h_regulatory:  promedio de errores homeostáticos   → sesga setpoints de B3
    h_adaptive:    traza de activaciones neuronales    → sesga inputs de B4
    h_heritable:   parámetros estables heredables      → se pasa a descendencia

    Mem = {h_structural, h_regulatory, h_adaptive, h_heritable, h_integrity, h_corruption_rate}
    """

    def __init__(self, genome: "Genome"):
        # Subtipos
        self.h_structural  = np.zeros(10)   # historia de boundary integrity (10 ticks)
        self.h_regulatory  = np.zeros(5)    # promedio suavizado de errores homeostáticos
        self.h_adaptive    = np.zeros(4)    # traza de activaciones neuronales (4 neuronas)
        self.h_heritable   = np.array([     # parámetros heredables estables
            genome.membrane_strength,
            genome.metabolic_base_rate,
            genome.conversion_efficiency,
            genome.repair_capacity_base,
            genome.waste_tolerance,
        ])

        # Integridad global de memoria
        self.h_integrity       = 1.0
        self.h_corruption_rate = 0.001

        # Costo de mantenimiento por tick (reducido — memoria cara pero no ruinosa)
        self.maintenance_cost_per_tick = 0.08  # ATP

        self._struct_ptr = 0  # puntero circular para h_structural

    def update(self, boundary_integrity: float, homeostatic_errors: Dict[str, float],
               neural_trace: np.ndarray, atp_available: float,
               waste_internal: float, damage: float) -> Dict:
        """
        Actualiza los 4 subtipos de memoria.
        Cuesta ATP. Degrada si no hay recursos (D4).
        """
        cost_paid = 0.0

        # ── D4: degradación de memoria por residuos y falta de reparación
        # Lenta pero real — memoria tiene un cuerpo material
        corruption_increment = waste_internal * 0.00008 + damage * 0.0002
        self.h_corruption_rate = min(0.015, self.h_corruption_rate + corruption_increment)
        self.h_integrity = max(0.0, self.h_integrity - self.h_corruption_rate * 0.3)

        # Si hay ATP, pagar mantenimiento y conservar memoria
        if atp_available >= self.maintenance_cost_per_tick:
            cost_paid = self.maintenance_cost_per_tick
            # decay de corrupción si bien mantenida
            self.h_corruption_rate = max(0.0005, self.h_corruption_rate - 0.002)
            self.h_integrity = min(1.0, self.h_integrity + 0.005)
        else:
            # sin mantenimiento → corrupción activa
            self.h_corruption_rate = min(0.015, self.h_corruption_rate + 0.001)

        # escala de actualización por integridad
        update_scale = self.h_integrity

        # ── h_structural: actualización circular
        noise_struct = np.random.randn() * self.h_corruption_rate * 0.1
        self.h_structural[self._struct_ptr] = boundary_integrity * update_scale + noise_struct
        self._struct_ptr = (self._struct_ptr + 1) % 10

        # ── h_regulatory: promedio exponencial de errores homeostáticos
        e_vec = np.array([
            homeostatic_errors.get('energy', 0),
            homeostatic_errors.get('damage', 0),
            homeostatic_errors.get('waste',  0),
            homeostatic_errors.get('boundary', 0),
            homeostatic_errors.get('memory', 0)
        ])
        noise_reg = np.random.randn(5) * self.h_corruption_rate * 0.05
        self.h_regulatory = (0.95 * self.h_regulatory + 0.05 * e_vec * update_scale + noise_reg)
        self.h_regulatory = np.clip(self.h_regulatory, -1, 1)

        # ── h_adaptive: traza de activaciones neuronales (B4 → B5)
        noise_adap = np.random.randn(4) * self.h_corruption_rate * 0.05
        self.h_adaptive = (0.90 * self.h_adaptive + 0.10 * neural_trace * update_scale + noise_adap)

        # ── h_heritable: degrada muy lentamente (D4/D5)
        noise_her = np.random.randn(5) * self.h_corruption_rate * 0.02
        self.h_heritable = np.clip(self.h_heritable + noise_her, 0.05, 1.5)

        return {
            "h_integrity": self.h_integrity,
            "corruption_rate": self.h_corruption_rate,
            "maintenance_cost": cost_paid
        }

    def repair_memory(self, repair_amount: float):
        """B6 repara memoria: reduce corrupción, restaura integridad (D4 reversal)."""
        self.h_corruption_rate = max(0.001, self.h_corruption_rate - repair_amount * 0.02)
        self.h_integrity       = min(1.0,   self.h_integrity        + repair_amount * 0.05)
        # pequeña restauración de h_heritable
        # (reparación parcial, no perfecta)
        self.h_heritable = np.clip(self.h_heritable, 0.05, 1.5)

    def get_structural_repair_signal(self) -> float:
        """
        B5 → B6: señal de memoria estructural para prioridad de reparación.
        Si la historia muestra frontera deteriorándose → urgir más reparación.
        """
        if np.all(self.h_structural == 0):
            return 0.5
        trend = np.mean(np.diff(self.h_structural)) if len(self.h_structural) > 1 else 0
        # tendencia negativa → señal alta de reparación
        return float(np.clip(0.5 - trend * 2.0, 0.0, 1.0))

    def get_regulatory_bias(self) -> np.ndarray:
        """B5 → B3: sesgo regulatorio desde memoria."""
        return self.h_regulatory.copy()

    def get_adaptive_bias(self) -> np.ndarray:
        """B5 → B4: sesgo adaptativo para cognición."""
        return self.h_adaptive.copy()

    def get_heritable_snapshot(self) -> np.ndarray:
        """H4/H1: parámetros heredables actuales (para reproducción)."""
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


# ─────────────────────────────────────────────────────────────
# B6: REPARACIÓN
# ─────────────────────────────────────────────────────────────

class RepairSystem:
    """
    B6: Sistema de reparación.
    Repara TODOS los bloques: frontera, metabolismo, cognitivo, memoria.
    Consume ATP y material reparativo.
    Coordinada por prioridades de B3.

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
        Distribuye presupuesto de reparación entre los 4 bloques.
        Retorna (damage_reduced, repair_log).
        """
        self._ticks_since_repair += 1
        if self._ticks_since_repair < self.p_latency:
            return 0.0, {}

        self._ticks_since_repair = 0

        # Presupuesto total: ATP disponible × prioridad de reparación
        atp_budget    = met.a_free * homeostasis.p_repair * 0.20
        struct_budget = met.p_repair * homeostasis.p_repair * 0.40

        if atp_budget < 0.5 or struct_budget < 0.5:
            return 0.0, {}

        atp_total    = met.consume_atp(atp_budget)
        struct_total = met.consume_repair_material(struct_budget)

        # Señal de memoria estructural → ajusta fracción para frontera
        struct_repair_signal = memory.get_structural_repair_signal()

        # ── Distribución de reparación entre bloques
        # Proporcional al daño de cada bloque + señal de memoria
        dmg_boundary = max(0, 1.0 - boundary.c_integrity)
        dmg_neural   = neural.t_noise_level
        dmg_memory   = 1.0 - memory.h_integrity
        dmg_metabolic = 1.0 - met.eta_metabolic

        total_need = dmg_boundary + dmg_neural + dmg_memory + dmg_metabolic + 1e-9
        # sesgo estructural desde B5
        boundary_extra = struct_repair_signal * 0.2

        frac_boundary = (dmg_boundary / total_need + boundary_extra)
        frac_neural   = dmg_neural   / total_need * (1 - boundary_extra)
        frac_memory   = dmg_memory   / total_need * (1 - boundary_extra)
        frac_metabolic= dmg_metabolic/ total_need * (1 - boundary_extra)

        # normaliza
        s = frac_boundary + frac_neural + frac_memory + frac_metabolic + 1e-9
        frac_boundary /= s; frac_neural /= s
        frac_memory   /= s; frac_metabolic /= s

        # ── Repara cada bloque
        atp_b, struct_b = boundary.repair(
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
# B7: REPRODUCCIÓN Y DESARROLLO
# ─────────────────────────────────────────────────────────────

class ReproductionModule:
    """
    B7: Reproducción y desarrollo.

    Condiciones (R1-R8): excedente metabólico, material estructural,
    material reproductivo, bajo daño, residuos controlados, memoria heredable íntegra,
    coherencia identitaria, madurez reproductiva.

    Herencia H1-H4: estructura, regulación, cognición, desarrollo.
    Fallos posibles: descendencia inviable/deformada, fallo completo con costo.
    """

    def __init__(self, genome: "Genome"):
        self.r_maturity   = 0.0   # crece con el tiempo
        self.r_material   = 0.0   # material reproductivo acumulado
        self.r_stability  = 1.0
        self.r_failure_risk = 0.0
        self.d_stage      = 0     # etapa de desarrollo del descendiente
        self.d_max_stage  = genome.development_ticks
        self._replicating = False
        self._abort_count = 0
        self._rng = random.Random()

    def tick_maturity(self, age: int, genome: "Genome", met: Metabolism,
                      damage: float):
        """Madura lentamente. Requiere mínima edad (H4)."""
        if age >= genome.repr_min_age:
            self.r_maturity = min(1.0, self.r_maturity + 0.012)
        # acumula material reproductivo desde metabolismo
        self.r_material = min(genome.reproductive_mass_cap,
                              self.r_material + met.q_repro * 0.3)
        # riesgo de fallo crece con daño acumulado (D5)
        self.r_failure_risk = min(0.8, damage * 0.6 + (1 - met.eta_metabolic) * 0.3)

    def can_reproduce(self, met: Metabolism, boundary: Boundary,
                      memory: MaterialMemory, homeostasis: HomeostasisRegulator,
                      damage: float, identity_I: float,
                      genome: "Genome") -> bool:
        """R1-R8: todas las condiciones deben cumplirse."""
        if self._replicating:
            return False
        # R1: excedente metabólico
        if met.atp_fraction < genome.repr_threshold_energy:      return False
        # R2: material estructural
        if met.m_struct < genome.structural_mass_cap * 0.40:     return False
        # R3: material reproductivo
        if self.r_material < 15.0:                                return False
        # R4: daño bajo
        if damage > genome.repr_threshold_damage:                 return False
        # R5: residuos controlados
        if met.waste_fraction > 0.45:                             return False
        # R6: memoria heredable íntegra
        if memory.h_integrity < 0.50:                             return False
        # R7: coherencia identitaria
        if identity_I < 0.40:                                     return False
        # R8: madurez reproductiva
        if self.r_maturity < 0.65:                                return False
        return True

    def initiate(self):
        self._replicating = True
        self.d_stage = 0

    def develop_tick(self, met: Metabolism, genome: "Genome") -> Optional[str]:
        """
        Avanza desarrollo del descendiente.
        Retorna: 'continue' | 'success' | 'fail' | 'abort'
        """
        if not self._replicating:
            return None

        # Costo de desarrollo por tick
        atp_cost = genome.maturation_cost_rate * met.a_free_cap * 0.03
        struct_cost = 1.0
        if met.a_free < atp_cost or met.m_struct < struct_cost:
            # abortar si no hay recursos
            self._abort_count += 1
            if self._abort_count > 5:
                self._replicating = False
                self._abort_count = 0
                # costo de aborto al progenitor
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
            # fallo por riesgo acumulado
            if self._rng.random() < self.r_failure_risk:
                return 'fail'
            return 'success'

        return 'continue'

    def build_offspring_genome(self, parent_genome: "Genome",
                                parent_neural: NeuralCore,
                                parent_memory: MaterialMemory,
                                rng: random.Random) -> Optional["Genome"]:
        """
        Construye genoma del descendiente con H1-H4.
        Puede retornar None si el material heredable está muy degradado.
        """
        if parent_memory.h_integrity < 0.3:
            # herencia tan corrupta que el descendiente es inviable
            return None

        # Base: mutación del genoma del progenitor
        child_genome = parent_genome.mutate(rng)

        # H1: herencia estructural — sesgada por h_heritable de B5
        h_her = parent_memory.get_heritable_snapshot()
        child_genome.membrane_strength     = float(np.clip(
            child_genome.membrane_strength * (0.7 + 0.3 * h_her[0]), 0.1, 1.0))
        child_genome.metabolic_base_rate   = float(np.clip(
            child_genome.metabolic_base_rate * (0.7 + 0.3 * h_her[1]), 0.1, 1.0))
        child_genome.conversion_efficiency = float(np.clip(
            child_genome.conversion_efficiency * (0.7 + 0.3 * h_her[2]), 0.1, 0.85))
        child_genome.repair_capacity_base  = float(np.clip(
            child_genome.repair_capacity_base * (0.7 + 0.3 * h_her[3]), 0.1, 1.0))

        # H2: herencia regulatoria — W_reg parcialmente del progenitor
        W_reg_parent = parent_genome.W_reg().flatten().tolist()
        W_reg_child  = child_genome.W_reg_flat
        child_genome.W_reg_flat = [
            0.60 * p + 0.40 * c
            for p, c in zip(W_reg_parent, W_reg_child)
        ]

        # H3: herencia cognitiva — predisposiciones neuronales (no aprendizaje completo)
        W_ih_pred, W_hh_pred, W_ho_pred = parent_neural.get_heritable_predispositions()
        child_genome.neural_W_ih_flat = [
            0.5 * p + 0.5 * c
            for p, c in zip(W_ih_pred, child_genome.neural_W_ih_flat)
        ]
        child_genome.neural_W_hh_flat = [
            0.5 * p + 0.5 * c
            for p, c in zip(W_hh_pred, child_genome.neural_W_hh_flat)
        ]
        child_genome.neural_W_ho_flat = [
            0.5 * p + 0.5 * c
            for p, c in zip(W_ho_pred, child_genome.neural_W_ho_flat)
        ]

        # H4: herencia de desarrollo — del genoma ya mutado
        # (tiempos de maduración, costos, umbrales — ya están en child_genome)

        # Integridad heredable: si baja → descendiente debilitado
        integrity_factor = parent_memory.h_integrity
        child_genome.membrane_strength     *= integrity_factor
        child_genome.repair_capacity_base  *= integrity_factor

        # Morfogenética: herencia 60% progenitor + 40% mutado (igual que H2 regulatoria)
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


# ─────────────────────────────────────────────────────────────
# B9: IDENTIDAD / INDIVIDUACIÓN
# ─────────────────────────────────────────────────────────────

class Identity:
    """
    B9: Identidad / individuación.
    No es módulo físico. Es condición sistémica computada de todos los bloques.

    I = f(boundary_continuity, memory_continuity, regulatory_coherence,
           structural_coherence, causal_closure_proxy)

    Si I < θI_dead → muerte organizacional M3.

    Id = {i_boundary_continuity, i_memory_continuity, i_regulatory_coherence,
          i_structural_coherence, i_causal_closure_proxy}
    """

    THETA_I_DEAD = 0.18   # umbral M3
    THETA_I_REP  = 0.40   # umbral para reproducción (R7)

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
        Calcula I como función ponderada de componentes de individuación.
        """
        # Continuidad de frontera
        self.i_boundary_continuity = boundary.c_integrity * (1 - boundary.c_permanent_damage)

        # Continuidad de memoria
        self.i_memory_continuity = memory.h_integrity

        # Coherencia regulatoria
        self.i_regulatory_coherence = homeostasis.g_regulatory_coherence

        # Coherencia estructural (metabolismo funcional)
        self.i_structural_coherence = met.eta_metabolic * (1 - min(1.0, damage))

        # Proxy de clausura organizacional:
        # ¿está el circuito cerrado? mide si todos los flujos están activos.
        #   M→A: metabolismo produce ATP
        f_met_to_atp  = min(1.0, met.atp_produced_last_tick / 3.0)
        #   A→Rep: reparación activa (usa ATP)
        f_atp_to_rep  = 1.0 if met.a_free > 5.0 and damage < 0.8 else 0.3
        #   Rep→B1: frontera siendo mantenida
        f_rep_to_bnd  = min(1.0, boundary.c_integrity / 0.5) if boundary.c_integrity > 0 else 0.0
        #   B1→M: frontera habilita intercambio
        f_bnd_to_met  = boundary.c_permeability_resource * boundary.c_integrity
        #   Cog→acción: neural activo
        f_cog_to_act  = min(1.0, neural.t_excitation / 0.1) if neural.t_excitation > 0.01 else 0.2

        # Proxy = media geométrica de los 5 flujos
        flows = [f_met_to_atp, f_atp_to_rep, f_rep_to_bnd, f_bnd_to_met, f_cog_to_act]
        product = 1.0
        for f in flows:
            product *= max(0.01, f)
        self.i_causal_closure_proxy = product ** (1.0/5)

        # ── I compuesta (ponderada)
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
        """M3: muerte organizacional — identidad colapsada irreversiblemente."""
        if len(self._history) < 10:
            return False
        # solo M3 si sostenidamente bajo el umbral
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
# B8b: COMUNICACIÓN BIOSEMIÓTICA INTERCELULAR
# ─────────────────────────────────────────────────────────────

class CommunicationSystem:
    """
    Comunicación intercelular para vida digital.

    No es NLP, no es embedding, no es mensajería humana. Es un sistema
    biosemiótico mínimo:

    1. Las células emiten señales costosas a un medio compartido.
    2. El mundo las difunde y degrada.
    3. Otras células las reciben según frontera + receptores heredables.
    4. La recepción modula homeostasis, cognición, movimiento y memoria.
    5. La señal solo "significa" algo porque altera supervivencia y conducta.

    Estado comunicativo:
      z_received    = señales locales percibidas
      z_decoded     = traducción corporal por receptor heredable
      z_social_move = sesgo vectorial de movimiento por gradientes de señal
      z_coherence   = estabilidad reciente de señal recibida
    """

    def __init__(self, genome: "Genome"):
        self.receptor_W = genome.signal_receptor_W().copy()
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
                homeostasis: HomeostasisRegulator,
                memory: MaterialMemory, damage: float) -> Dict:
        """
        Lee señales locales + gradientes y las traduce a efectos corporales.

        La frontera regula entrada de señales; memoria dañada y ruido interno degradan
        comprensión. La saturación tanh impide que una señal infinita controle todo.
        """
        raw = world.sample_signals(x, y)
        permeability = boundary.c_permeability_signal * boundary.c_integrity
        memory_gate = 0.4 + 0.6 * memory.h_integrity
        metabolic_gate = 0.35 + 0.65 * met.atp_fraction
        received = raw * permeability * self.sensitivity * memory_gate * metabolic_gate

        # Normalización biofísica: señales fuertes saturan, no escalan infinito.
        norm = np.tanh(received / 18.0)
        self.z_received = norm
        self.z_signal_load = float(np.mean(norm))

        # Traducción heredable a cuatro ejes corporales:
        # [mantenimiento, reparación, acción, reproducción].
        decoded = np.tanh(self.receptor_W @ norm)
        self.z_decoded = decoded

        # Gradientes: la célula no solo "oye" intensidad; detecta dirección.
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

        # Sesgo de prioridades: lo social no reemplaza homeostasis; la perturba.
        # Orden: [maintenance, repair, action, reproduction].
        self.z_priority_bias = np.array([
            0.08 * norm[SIGNAL_IDX["energy_need"]] + 0.04 * norm[SIGNAL_IDX["death_trace"]],
            0.10 * norm[SIGNAL_IDX["repair_need"]] + 0.08 * norm[SIGNAL_IDX["toxin_alarm"]],
            0.07 * norm[SIGNAL_IDX["nutrient_beacon"]] - 0.05 * norm[SIGNAL_IDX["crowding"]],
            0.06 * norm[SIGNAL_IDX["reproduction_ready"]] - 0.04 * norm[SIGNAL_IDX["death_trace"]],
        ], dtype=np.float64)
        self.z_priority_bias += decoded * 0.03
        self.z_priority_bias = np.clip(self.z_priority_bias, -0.15, 0.20)

        # Coherencia = estabilidad de lo recibido; si cambia violentamente,
        # la célula no "entiende" con seguridad.
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
        Integra señales sociales en B3 sin sobrescribir la red regulatoria.
        Es una perturbación pequeña y normalizada, no un if externo dominante.
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
        Devuelve sesgos para los tres inputs ambientales de NeuralCore:
        nutriente, toxina, daño. Mantiene dimensionalidad 6 para no romper B4.
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
        Emite señales ancladas al estado vital. Emitir cuesta ATP, así que no es
        telepatía gratuita ni broadcast artificial.
        """
        v = np.zeros(N_SIGNAL_CHANNELS, dtype=np.float64)

        atp = met.atp_fraction
        waste = met.waste_fraction

        # Señales positivas: solo si la célula está estable, para evitar mentiras
        # accidentales desde organismos colapsados.
        stability = boundary.c_integrity * memory.h_integrity * identity.I
        v[SIGNAL_IDX["nutrient_beacon"]] = max(0.0, min(1.0, nut_local / 80.0)) * stability

        # Alarmas y necesidades: salen de errores homeostáticos reales.
        v[SIGNAL_IDX["toxin_alarm"]] = np.clip(tox_local / 40.0 + waste * 0.7 + damage * 0.5, 0, 1)
        v[SIGNAL_IDX["energy_need"]] = np.clip(homeostasis.g_energy_error + max(0, 0.28 - atp), 0, 1)
        v[SIGNAL_IDX["repair_need"]] = np.clip(homeostasis.g_damage_error + homeostasis.g_boundary_error +
                                               (1.0 - memory.h_integrity) * 0.35, 0, 1)

        v[SIGNAL_IDX["reproduction_ready"]] = np.clip(
            reproduction.r_maturity * (1.0 - damage) * identity.I *
            (1.0 if not reproduction._replicating else 0.25),
            0, 1
        )

        # Célula viva ocupa espacio: comunica saturación ecológica local.
        v[SIGNAL_IDX["crowding"]] = 0.15 + 0.35 * self.z_received[SIGNAL_IDX["crowding"]]

        # Células moribundas emiten traza de colapso antes de morir.
        dying = 1.0 if phase in (LifePhase.DYING, LifePhase.STRESSED) else 0.0
        v[SIGNAL_IDX["death_trace"]] = np.clip(dying * (1.0 - identity.I + damage), 0, 1)

        # Selectividad: reduce ruido de señales débiles; linajes pueden ser más
        # "parlantes" o más "reservados".
        threshold = 0.08 + (1.0 - min(1.0, self.selectivity)) * 0.10
        v[v < threshold] = 0.0

        # Costo energético proporcional a carga emitida.
        load = float(np.sum(v))
        cost = load * self.cost_factor * (1.0 + 0.5 * homeostasis.g_stress)
        paid = met.consume_atp(cost)
        if cost > 1e-9:
            v *= min(1.0, paid / cost)

        # Depósito local. Los multiplicadores convierten [0,1] en concentración.
        for i, amount in enumerate(v):
            if amount > 0:
                world.deposit_signal(x, y, i, amount * self.emission_strength * 8.0)

        self.z_emitted_last = v
        return {"emitted": v, "cost": cost, "paid": paid}

    def repair(self, repair_amount: float):
        """B6 también puede estabilizar receptores comunicativos."""
        self.z_coherence = min(1.0, self.z_coherence + repair_amount * 0.04)
        self.receptor_W = np.clip(self.receptor_W, -3, 3)

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
# FASE DE VIDA
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
    Relación persistente entre dos células.

    Esto es el primer objeto que convierte señales de campo en topología
    biológica: vínculo con costo, estabilidad, capacidad y memoria mínima.
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
            "queue": len(self.event_queue),
        }


@dataclass
class OrganismState:
    """
    Estado causal de la organización multicelular.

    No reemplaza a las células. Mide si existe una unidad superior con
    topología, intercambio, roles y presión regulatoria propia.
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

    def update(self, cells: Dict[str, "Cell"], junctions: Dict[str, Junction]):
        alive = [c for c in cells.values() if c.alive]
        self.member_cell_ids = {c.id for c in alive}
        self.junction_ids = {jid for jid, j in junctions.items()
                             if j.cell_a in self.member_cell_ids and j.cell_b in self.member_cell_ids}

        if not alive:
            self.shared_stress = 0.0
            self.boundary_integrity = 0.0
            self.collective_energy_pressure = 1.0
            self.collective_damage = 1.0
            self.collective_identity = 0.0
            self.development_stage = "extinct"
            self.reproduction_pressure = 0.0
            self.role_coverage = 0.0
            self.topology_integrity = 0.0
            return

        n = len(alive)
        self.shared_stress = float(np.mean([c.homeostasis.g_stress for c in alive]))
        self.collective_energy_pressure = float(np.mean([1.0 - c.metabolism.atp_fraction for c in alive]))
        self.collective_damage = float(np.mean([c.damage_X for c in alive]))

        boundary_cells = [c for c in alive if c.cell_type == CellType.BOUNDARY]
        self.boundary_integrity = float(np.mean([c.boundary.c_integrity for c in boundary_cells])
                                        if boundary_cells else 0.0)

        active_junctions = [junctions[jid] for jid in self.junction_ids]
        adhesion_like = [j for j in active_junctions if j.kind in (JunctionKind.ADHESION, JunctionKind.GAP,
                                                                   JunctionKind.METABOLIC, JunctionKind.SYNAPTIC)]
        possible_edges = max(1, n - 1)
        self.topology_integrity = float(np.clip(
            len(adhesion_like) / possible_edges *
            (np.mean([j.strength for j in adhesion_like]) if adhesion_like else 0.0),
            0.0, 1.0
        ))

        required_roles = {
            CellType.BOUNDARY, CellType.METABOLIC, CellType.REPAIR,
            CellType.SIGNALING, CellType.NEURON
        }
        present_roles = {c.cell_type for c in alive}
        self.role_coverage = len(required_roles & present_roles) / len(required_roles)

        mean_identity = float(np.mean([c.identity.I for c in alive]))
        exchange = float(np.clip(
            sum(j.transport_capacity * j.strength for j in active_junctions
                if j.kind in (JunctionKind.GAP, JunctionKind.METABOLIC)) / max(1.0, n),
            0.0, 1.0
        ))
        self.collective_identity = float(np.clip(
            0.28 * mean_identity +
            0.22 * self.topology_integrity +
            0.18 * exchange +
            0.18 * self.role_coverage +
            0.14 * (1.0 - self.shared_stress),
            0.0, 1.0
        ))

        germline_ready = any(c.cell_type == CellType.GERMLINE and
                             c.reproduction.r_maturity > 0.7 and c.identity.I > 0.55
                             for c in alive)
        self.reproduction_pressure = float(np.clip(
            (1.0 if germline_ready else 0.0) *
            self.collective_identity *
            (1.0 - self.collective_energy_pressure),
            0.0, 1.0
        ))

        if n < 3 or not active_junctions:
            self.development_stage = "solitary"
        elif self.collective_identity < 0.35:
            self.development_stage = "aggregate"
        elif self.role_coverage < 0.6:
            self.development_stage = "proto_tissue"
        else:
            self.development_stage = "integrated"

    def to_dict(self) -> Dict:
        return {
            "id": self.organism_id,
            "members": len(self.member_cell_ids),
            "junctions": len(self.junction_ids),
            "shared_stress": round(self.shared_stress, 3),
            "boundary_integrity": round(self.boundary_integrity, 3),
            "energy_pressure": round(self.collective_energy_pressure, 3),
            "collective_damage": round(self.collective_damage, 3),
            "collective_identity": round(self.collective_identity, 3),
            "development_stage": self.development_stage,
            "reproduction_pressure": round(self.reproduction_pressure, 3),
            "role_coverage": round(self.role_coverage, 3),
            "topology_integrity": round(self.topology_integrity, 3),
        }


# ─────────────────────────────────────────────────────────────
# CÉLULA — integra los 9 bloques
# ─────────────────────────────────────────────────────────────

class Cell:
    """
    CNDV: Célula Neuronal Digital Viva.

    Integra B1-B9 en tick-order estricto del documento arquitectónico:
    1. intercambio con entorno        (B1 + B8)
    2. transformación metabólica      (B2)
    3. acumulación de residuo y daño  (D1-D7)
    4. actualización homeostática     (B3)
    5. reparación priorizada          (B6)
    6. cognición y acción             (B4)
    7. actualización de memoria       (B5)
    8. evaluación de viabilidad       (B9)
    9. evaluación de reproducción     (B7)
    10. muerte si organización no recuperable (M1, M2, M3)
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

        # ── 9 Bloques
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

        # Estado global S(t)
        self.damage_X    = 0.0   # daño acumulado total
        self.age_ticks   = 0
        self.phase       = LifePhase.DEVELOPING if developing else LifePhase.ACTIVE
        self.alive       = True
        self.death_cause: Optional[str] = None
        self.cell_type   = CellType.STEM if developing else CellType.METABOLIC
        self.type_commitment = 0.15 if developing else 0.35

        # Capa multicelular: vínculos persistentes, cooperación y vigilancia.
        self.junction_ids: Set[str] = set()
        self.attachment_strength = 0.0
        self.synaptic_input = np.zeros(4, dtype=np.float64)
        self.last_spike_tick: Optional[int] = None
        self.spike_output = 0.0
        self._refractory_ticks = 0
        self.contribution_score = 0.0
        self.received_support = 0.0
        self.provided_support = 0.0
        self.cheater_score = 0.0
        self.unauthorized_reproduction_attempts = 0

        # Historial
        self.event_log: List[str] = []
        self.metrics_history: List[Dict] = []
        self._starving_ticks = 0
        self._generation = 0

        world.register(x, y, self.id)

    # ─────────────────────────────────────────────────────────
    # TICK PRINCIPAL
    # ─────────────────────────────────────────────────────────

    def tick(self) -> Optional["Cell"]:
        """
        Ejecuta un tick completo en el orden del documento arquitectónico.
        Retorna una célula hija si hay reproducción exitosa, None si no.
        """
        if not self.alive:
            return None

        self.age_ticks += 1
        offspring = None

        # ──────────────────────────────────────────────────────
        # PASO 1: Intercambio con entorno (B1 + B8)
        # ──────────────────────────────────────────────────────
        nut_local, tox_local = self.world.sample(self.x, self.y)

        # Recepción comunicativa situada: la célula percibe el campo social antes
        # de decidir captura, reparación y movimiento.
        comm_in = self.communication.receive(
            self.world, self.x, self.y,
            self.boundary, self.metabolism, self.homeostasis,
            self.memory, self.damage_X
        )

        # La frontera filtra lo que entra
        # Recurso que entra: modulado por integridad + capacidad de transporte + neural_gate
        capture_base = (nut_local * self.boundary.c_permeability_resource *
                        self.boundary.c_transport_capacity)

        # El módulo neuronal contribuirá su capture_modulation después del paso 6,
        # pero usamos la última señal disponible (del tick anterior)
        capture_mod = float((self.neural.t_action_bias[2] + 1) / 2) if np.any(
            self.neural.t_action_bias != 0) else 0.5

        # Captura real — sin doble multiplicación de transport
        social_capture = 1.0 + 0.18 * self.communication.z_received[SIGNAL_IDX["nutrient_beacon"]]
        resource_captured = min(nut_local * 0.60, capture_base * (0.6 + 0.4 * capture_mod) * social_capture)
        resource_captured = self.world.consume_nutrient(self.x, self.y, resource_captured)
        self.metabolism.r_raw = min(self.metabolism.r_raw_cap,
                                    self.metabolism.r_raw + resource_captured)

        # Toxinas que entran a través de frontera dañada
        tox_entering = tox_local * self.boundary.c_permeability_toxin
        self.metabolism.w_waste = min(
            self.metabolism.w_waste_cap,
            self.metabolism.w_waste + tox_entering * 0.3
        )

        # ──────────────────────────────────────────────────────
        # PASO 2: Transformación metabólica (B2)
        # ──────────────────────────────────────────────────────
        met_out = self.metabolism.step(
            self.genome,
            tox_internal=self.metabolism.w_waste,
            maintenance_priority=self.homeostasis.p_maintenance
        )

        # Excreción de residuos al entorno
        waste_excreted = self.metabolism.w_waste * 0.04 * self.boundary.c_integrity
        self.metabolism.w_waste -= waste_excreted
        self.world.deposit_toxin(self.x, self.y, waste_excreted * 0.5)

        # ──────────────────────────────────────────────────────
        # PASO 3: Acumulación de daño (D1-D7)
        # ──────────────────────────────────────────────────────
        # D1: Frontera
        dmg_boundary = self.boundary.degrade(
            tox_local, self.metabolism.w_waste, self.age_ticks, self.genome
        )

        # D2: Metabólico — ya aplicado dentro de metabolism.step()

        # D6: Ecológico — escasez local degrada gradualmente
        if nut_local < 5.0:
            self.damage_X = min(1.0, self.damage_X + 0.003)

        # D7: Organizacional — si clausura rota, se añade daño difuso
        if self.identity.i_causal_closure_proxy < 0.3:
            self.damage_X = min(1.0, self.damage_X + 0.005)

        # Daño total acumulado
        self.damage_X = min(1.0, self.damage_X + dmg_boundary * 0.5)

        # Costo de mantenimiento de frontera
        boundary_maint = self.metabolism.consume_atp(
            self.boundary.c_maintenance_cost * self.metabolism.a_free_cap * 0.01
        )

        # ──────────────────────────────────────────────────────
        # PASO 4: Actualización homeostática (B3)
        # ──────────────────────────────────────────────────────
        mem_reg_bias = self.memory.get_regulatory_bias()  # B5 → B3
        hom_out = self.homeostasis.update(
            self.metabolism, self.boundary,
            self.damage_X, self.memory.h_integrity,
            mem_reg_bias
        )
        # Plasticidad regulatoria: W_reg aprende lentamente
        self.homeostasis.adapt_W_reg(
            learning_signal=self.memory.h_integrity * 0.1
        )
        # Señales sociales sesgan suavemente la homeostasis, sin reemplazarla.
        self.communication.modulate_homeostasis(self.homeostasis)

        # ──────────────────────────────────────────────────────
        # PASO 5: Reparación priorizada (B6)
        # ──────────────────────────────────────────────────────
        dmg_reduced, rep_log = self.repair.execute(
            self.metabolism, self.boundary, self.neural, self.memory,
            self.homeostasis, self.damage_X, self.genome
        )
        self.damage_X = max(0.0, self.damage_X - dmg_reduced)
        if dmg_reduced > 0:
            self.communication.repair(dmg_reduced)

        # ──────────────────────────────────────────────────────
        # PASO 6: Cognición y acción (B4)
        # ──────────────────────────────────────────────────────
        # Inputs normalizados del estado interno y entorno
        sig_nut, sig_tox, sig_damage = self.communication.neural_environment_bias(
            nut_local, tox_local, self.damage_X
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
            # Entrada intercelular directa. Se inyecta como sesgo interno, no como
            # nutriente/toxina fingidos, para mantener separadas fisiología y sinapsis.
            adaptive_bias = adaptive_bias + np.clip(self.synaptic_input, -1.0, 1.0)
            self.synaptic_input *= 0.35

        neural_out = self.neural.step(
            neural_inputs,
            adaptive_bias,
            self.metabolism.a_free
        )
        if self._refractory_ticks > 0:
            self.spike_output = 0.0
            self._refractory_ticks -= 1
        elif self.neural.t_excitation > EXCIT_THRESHOLD:
            self.spike_output = float(np.clip(
                (self.neural.t_excitation - EXCIT_THRESHOLD) * 3.0, 0.0, 1.0))
            self._refractory_ticks = REFRACTORY_PERIOD
        else:
            self.spike_output = 0.0

        # Acción: movimiento guiado por neuronal + quimiotaxis (B8)
        self._move(neural_out, nut_local, tox_local)

        # Modulación de frontera por cognición (F3: B4 → B1)
        gate_signal = float((neural_out['capture_modulation'] +
                            self.boundary.c_integrity) / 2)
        self.boundary.apply_neural_gate(gate_signal)

        # Costo cognitivo (real pero contenido)
        self.metabolism.consume_atp(
            self.neural.t_excitation * 0.3 * self.homeostasis.p_action
        )

        # ──────────────────────────────────────────────────────
        # PASO 7: Actualización de memoria (B5)
        # ──────────────────────────────────────────────────────
        mem_out = self.memory.update(
            boundary_integrity=self.boundary.c_integrity,
            homeostatic_errors=hom_out.get('errors', {}),
            neural_trace=self.neural.t_adaptive_trace,
            atp_available=self.metabolism.a_free,
            waste_internal=self.metabolism.w_waste,
            damage=self.damage_X
        )
        # Pagar costo de mantenimiento de memoria
        self.metabolism.consume_atp(mem_out.get('maintenance_cost', 0))

        # ──────────────────────────────────────────────────────
        # PASO 8: Evaluación de viabilidad (B9)
        # ──────────────────────────────────────────────────────
        I = self.identity.compute(
            self.boundary, self.metabolism, self.memory,
            self.homeostasis, self.neural, self.damage_X
        )

        # Actualizar fase según estado
        self._update_phase()

        # Emisión comunicativa: ocurre después de computar identidad para que
        # la señal enviada refleje el estado organizacional actual.
        comm_out = self.communication.emit(
            self.world, self.x, self.y,
            self.metabolism, self.boundary, self.homeostasis,
            self.memory, self.reproduction, self.identity,
            self.damage_X, nut_local, tox_local, self.phase
        )

        # ──────────────────────────────────────────────────────
        # PASO 9: Evaluación de reproducción (B7)
        # ──────────────────────────────────────────────────────
        self.reproduction.tick_maturity(self.age_ticks, self.genome,
                                        self.metabolism, self.damage_X)

        if not self.reproduction._replicating:
            if self.reproduction.can_reproduce(
                self.metabolism, self.boundary, self.memory,
                self.homeostasis, self.damage_X, I, self.genome
            ):
                self.reproduction.initiate()
                self._log("REPRODUCTION_INITIATED")

        if self.reproduction._replicating:
            result = self.reproduction.develop_tick(self.metabolism, self.genome)
            if result == 'success':
                offspring = self._spawn_offspring()
                if offspring:
                    self._log(f"OFFSPRING:{offspring.id}")
                else:
                    self._log("OFFSPRING_FAILED:genome_corrupted")
            elif result == 'fail':
                self._log("REPRODUCTION_FAILED:risk_exceeded")
            elif result == 'abort':
                self._log("REPRODUCTION_ABORTED:no_resources")

        # ──────────────────────────────────────────────────────
        # PASO 10: Verificación de muerte (M1, M2, M3)
        # ──────────────────────────────────────────────────────
        self._check_death()

        # Registro de métricas
        self._record_metrics()

        return offspring

    # ─────────────────────────────────────────────────────────
    # MOVIMIENTO (B8)
    # ─────────────────────────────────────────────────────────

    def _move(self, neural_out: Dict, nut_local: float, tox_local: float):
        """
        B8: Movimiento guiado por output neuronal + quimiotaxis.
        La célula integra señales propias (B4) con gradientes del mundo.
        """
        if self.metabolism.a_free < 2.0:
            return  # sin energía no se mueve

        sr = self.genome.sensor_range

        # Gradientes de nutrientes y toxinas
        dnut_x, dnut_y = self.world.gradient(self.x, self.y, self.world.nutrients)
        dtox_x, dtox_y = self.world.gradient(self.x, self.y, self.world.toxins)

        # Señal neural de movimiento (B4 output)
        neural_x = neural_out['move_x'] * self.genome.motility
        neural_y = neural_out['move_y'] * self.genome.motility

        # Quimiotaxis: atracción a nutrientes, repulsión de toxinas
        chemotaxis_x = (dnut_x * self.genome.chemotaxis_gain -
                        dtox_x * self.genome.toxin_avoidance)
        chemotaxis_y = (dnut_y * self.genome.chemotaxis_gain -
                        dtox_y * self.genome.toxin_avoidance)

        # Comunicación: gradientes de señales de otras células.
        # No domina; orienta cuando hay recurso, alarma, muerte o saturación social.
        social_x, social_y = self.communication.z_social_move

        # Integración: neuronal domina, quimiotaxis y comunicación como sesgos vivos
        total_x = 0.52 * neural_x + 0.30 * chemotaxis_x + 0.18 * social_x
        total_y = 0.52 * neural_y + 0.30 * chemotaxis_y + 0.18 * social_y
        if self.attachment_strength > 0.0:
            # Una célula adherida ya no es una partícula libre. La adhesión no
            # implementa movimiento colectivo completo, pero sí hace costoso y
            # raro romper una topología estable por desplazamiento individual.
            drag = np.clip(self.attachment_strength, 0.0, 0.9)
            total_x *= (1.0 - 0.55 * drag)
            total_y *= (1.0 - 0.55 * drag)
            if self.attachment_strength > 0.72 and self.rng.random() > 0.06:
                return

        # Normaliza dirección
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

        # Costo de movimiento (reducido)
        move_cost = 0.15 + self.genome.motility * 0.1 + self.attachment_strength * 0.25
        actual = self.metabolism.consume_atp(move_cost)
        if actual < move_cost * 0.5:
            return  # no alcanza el ATP

        if not self.world.is_occupied(new_x, new_y):
            self.world.unregister(self.x, self.y)
            self.x = new_x
            self.y = new_y
            self.world.register(self.x, self.y, self.id)

    # ─────────────────────────────────────────────────────────
    # DESCENDENCIA (B7 spawn)
    # ─────────────────────────────────────────────────────────

    def _spawn_offspring(self) -> Optional["Cell"]:
        """Crea y posiciona descendiente. La reproducción no copia estado completo."""
        child_genome = self.reproduction.build_offspring_genome(
            self.genome, self.neural, self.memory, self.rng
        )
        if child_genome is None:
            return None

        # Buscar posición adyacente libre
        candidates = [(self.x + dx, self.y + dy)
                      for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                      if (dx, dy) != (0, 0)]
        self.rng.shuffle(candidates)

        for cx, cy in candidates:
            cx = cx % self.world.W; cy = cy % self.world.H
            if not self.world.is_occupied(cx, cy):
                # Costo al progenitor
                self.metabolism.consume_atp(4.0)
                self.metabolism.consume_structural(2.5)
                self.damage_X = min(1.0, self.damage_X + 0.04)

                child = Cell(cx, cy, self.world, child_genome,
                             rng=random.Random(self.rng.randint(0, 2**31)),
                             developing=True)
                child._generation = self._generation + 1
                return child

        return None

    # ─────────────────────────────────────────────────────────
    # VERIFICACIÓN DE MUERTE
    # ─────────────────────────────────────────────────────────

    def _check_death(self):
        """
        M1: Muerte metabólica
        M2: Muerte estructural
        M3: Muerte organizacional
        """
        # M1: energía críticamente baja por tiempo sostenido
        if self.metabolism.is_starving:
            self._starving_ticks += 1
            if self._starving_ticks > 15:
                self._die("M1_metabolica")
                return
        else:
            self._starving_ticks = max(0, self._starving_ticks - 1)

        # M1b: residuos tóxicos terminales
        if self.metabolism.w_waste > self.metabolism.w_waste_cap * 0.90:
            self._die("M1_intoxicacion")
            return

        # M2: colapso estructural de frontera
        if self.boundary.is_dead:
            self._die("M2_structural")
            return

        # M2b: daño sistémico irrecuperable
        if self.damage_X > 0.95:
            self._die("M2_damage")
            return

        # M3: muerte organizacional (identidad colapsada sostenidamente)
        if self.identity.is_organizationally_dead:
            self._die("M3_organizacional")
            return

    def _die(self, cause: str):
        self.alive = False
        self.death_cause = cause
        self.phase = LifePhase.DEAD
        # necroseñal: la muerte también informa al medio.
        if hasattr(self.world, "deposit_signal"):
            self.world.deposit_signal(self.x, self.y, "death_trace", 35.0)
        self.world.unregister(self.x, self.y)
        # devuelve algo de nutrientes al entorno (materia no se destruye)
        self.world.deposit_nutrient(self.x, self.y,
            self.metabolism.r_raw * 0.4 + self.metabolism.m_struct * 0.2)
        self._log(f"DEAD:{cause}")

    # ─────────────────────────────────────────────────────────
    # FASE Y LOGS
    # ─────────────────────────────────────────────────────────

    def _update_phase(self):
        if not self.alive:
            return
        I = self.identity.I
        if self.age_ticks < self.genome.development_ticks:
            self.phase = LifePhase.DEVELOPING
        elif self.reproduction._replicating:
            self.phase = LifePhase.REPLICATING
        elif self.homeostasis.g_stress > 0.5 or self.damage_X > 0.4:
            self.phase = LifePhase.STRESSED
        elif self.damage_X > 0.2 or self.boundary.c_integrity < 0.6:
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
                "damage": round(self.damage_X, 3),
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
            # Estado S(t)
            "S": {
                "C": round(self.boundary.c_integrity, 3),
                "A": round(self.metabolism.atp_fraction, 3),
                "M": round(self.metabolism.m_struct, 2),
                "P": round(self.metabolism.p_repair, 2),
                "W": round(self.metabolism.waste_fraction, 3),
                "X": round(self.damage_X, 3),
                "I": round(self.identity.I, 3),
            },
            # Bloques
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
                "attachment_strength": round(self.attachment_strength, 3),
                "spike_output": round(self.spike_output, 3),
                "contribution": round(self.contribution_score, 3),
                "provided_support": round(self.provided_support, 3),
                "received_support": round(self.received_support, 3),
                "cheater_score": round(self.cheater_score, 3),
            },
            # Historial
            "events": self.event_log[-10:],
            "metrics_history": self.metrics_history[-10:]
        }


# ─────────────────────────────────────────────────────────────
# COLONIA
# ─────────────────────────────────────────────────────────────

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
            # Spawn cerca de fuentes de nutrientes
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
                # Recursos iniciales más generosos para células primordiales
                cell.metabolism.r_raw  = 50.0
                cell.metabolism.a_free = 60.0
                self.cells[cell.id] = cell
                placed += 1

    def spawn_primordial_from_genome(self, n: int, genome: "Genome"):
        """
        Como spawn_primordial pero todas las células descienden de un genoma fundador.
        Usado por EvolutionEngine para sembrar colonias de reemplazo con el genoma ganador.
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
        self._maintain_junctions_and_share()
        self._update_attachment_strengths()

        # Tick todas las células vivas
        to_add: List[Cell] = []
        to_remove: List[str] = []

        for cid, cell in list(self.cells.items()):
            if not cell.alive:
                to_remove.append(cid)
                continue
            offspring = cell.tick()
            if offspring and len(self.cells) + len(to_add) < self.max_cells:
                to_add.append(offspring)
            if not cell.alive:
                to_remove.append(cid)

        for cid in to_remove:
            dead = self.cells.pop(cid, None)
            if dead:
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

        for cell in to_add:
            self.cells[cell.id] = cell

        self._form_multicellular_links()
        self._differentiate_cells()
        self._queue_synaptic_events()
        self._police_cells()
        self.organism.update(self.cells, self.junctions)
        self._apply_organism_pressure()
        self._try_organism_reproduction()

        if include_status:
            return self.status()
        return None

    # ─────────────────────────────────────────────────────────
    # CAPA MULTICELULAR
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
        for jid in list(self.junctions.keys()):
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
        vitality = 0.5 * (a.identity.I + b.identity.I) * (1.0 - 0.5 * (a.damage_X + b.damage_X))
        phase_ok = 0.85 if LifePhase.DEAD not in (a.phase, b.phase) else 0.0
        return float(np.clip(0.45 * kinship + 0.45 * vitality + 0.10 * phase_ok, 0.0, 1.0))

    def _form_multicellular_links(self):
        if self.tick_count % 3 != 0:
            return
        alive = [c for c in self.cells.values() if c.alive]
        if len(alive) < 2:
            return

        for cell in alive:
            if len(cell.junction_ids) >= 8:
                continue
            neighbors = self.world.get_neighbor_cells(cell.x, cell.y, 1, self.cells)
            self.rng.shuffle(neighbors)
            for nb in neighbors[:4]:
                other = self.cells.get(nb["id"])
                if other is None or len(other.junction_ids) >= 8:
                    continue
                if not self._has_junction(JunctionKind.ADHESION, cell.id, other.id):
                    comp = self._compatibility(cell, other)
                    crowd = cell.communication.z_received[SIGNAL_IDX["crowding"]]
                    if comp > 0.48 and crowd < 0.75 and self.rng.random() < comp * 0.16:
                        self._create_junction(
                            JunctionKind.ADHESION, cell.id, other.id,
                            strength=0.22 + 0.45 * comp,
                            transport=0.03,
                            conductance=0.08
                        )
                        break

        for j in list(self.junctions.values()):
            if j.kind != JunctionKind.ADHESION or j.age < 6 or j.strength < 0.35:
                continue
            a = self.cells.get(j.cell_a); b = self.cells.get(j.cell_b)
            if a is None or b is None:
                continue
            if not self._has_junction(JunctionKind.METABOLIC, a.id, b.id):
                need_gap = abs(a.metabolism.atp_fraction - b.metabolism.atp_fraction)
                repair_gap = abs(a.damage_X - b.damage_X)
                if need_gap + repair_gap > 0.18 and self.rng.random() < 0.18:
                    self._create_junction(
                        JunctionKind.METABOLIC, a.id, b.id,
                        strength=j.strength * 0.65,
                        transport=0.10 + 0.40 * j.strength,
                        conductance=0.18
                    )
            if not self._has_junction(JunctionKind.SYNAPTIC, a.id, b.id):
                neural_pair = (
                    a.cell_type in (CellType.NEURON, CellType.SENSORY, CellType.MOTOR) or
                    b.cell_type in (CellType.NEURON, CellType.SENSORY, CellType.MOTOR)
                )
                if neural_pair and self.rng.random() < 0.12:
                    # Stable pre/post by UUID order (not instantaneous excitation)
                    pre, post = (a, b) if a.id < b.id else (b, a)
                    syn = self._create_junction(
                        JunctionKind.SYNAPTIC, pre.id, post.id,
                        strength=j.strength * 0.55,
                        transport=0.0,
                        conductance=0.20 + 0.45 * j.strength,
                        weight=0.18 + 0.45 * j.strength
                    )
                    # ~30% inhibitory synapses to enable E/I balance
                    if syn is not None and self.rng.random() < 0.30:
                        syn.receptor_type = "inhibitory"
                        syn.weight = -abs(syn.weight)

    def _maintain_junctions_and_share(self):
        for j in list(self.junctions.values()):
            a = self.cells.get(j.cell_a); b = self.cells.get(j.cell_b)
            if a is None or b is None or not a.alive or not b.alive:
                self._remove_junction(j.id)
                continue

            j.age += 1
            distance = self._torus_distance(a, b)
            max_distance = 4.0 if j.kind == JunctionKind.SYNAPTIC else 2.0
            if distance > max_distance:
                j.strength = max(0.0, j.strength - 0.08 * (distance - max_distance))
                j.stability = max(0.0, j.stability - 0.04 * (distance - max_distance))
                if distance > max_distance * 1.8:
                    self._remove_junction(j.id)
                    continue

            type_discount = 0.75 if (a.cell_type == CellType.BOUNDARY or b.cell_type == CellType.BOUNDARY) else 1.0
            due = j.maintenance_cost * type_discount
            paid = a.metabolism.consume_atp(due * 0.5) + b.metabolism.consume_atp(due * 0.5)
            if paid + 1e-9 < due * 0.65:
                j.stability = max(0.0, j.stability - 0.035)
                j.strength = max(0.0, j.strength - 0.025)
            else:
                j.stability = min(1.0, j.stability + 0.006)
                j.strength = min(1.0, j.strength + 0.004 * j.stability)

            if j.kind in (JunctionKind.METABOLIC, JunctionKind.GAP):
                self._resource_share(a, b, j)

            idle = self.tick_count - j.last_activity
            if j.kind == JunctionKind.SYNAPTIC and idle > 80:
                j.prune_score += 0.015
            if j.strength < 0.04 or j.stability < 0.03 or j.prune_score > 1.0:
                self._remove_junction(j.id)

    def _resource_share(self, a: Cell, b: Cell, j: Junction):
        cap = j.transport_capacity * j.strength
        if cap <= 0:
            return

        def move_resource(attr: str, cap_attr: str, donor: Cell, receiver: Cell,
                          amount: float, support_weight: float):
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

        if a.metabolism.atp_fraction > b.metabolism.atp_fraction + 0.16 and a.metabolism.atp_fraction > 0.35:
            move_resource("a_free", "a_free_cap", a, b, cap * 2.2, 1.0)
        elif b.metabolism.atp_fraction > a.metabolism.atp_fraction + 0.16 and b.metabolism.atp_fraction > 0.35:
            move_resource("a_free", "a_free_cap", b, a, cap * 2.2, 1.0)

        if a.metabolism.r_raw > b.metabolism.r_raw + 16:
            move_resource("r_raw", "r_raw_cap", a, b, cap * 1.6, 0.45)
        elif b.metabolism.r_raw > a.metabolism.r_raw + 16:
            move_resource("r_raw", "r_raw_cap", b, a, cap * 1.6, 0.45)

        if a.metabolism.p_repair > b.metabolism.p_repair + 8 and b.damage_X > a.damage_X:
            move_resource("p_repair", "p_repair_cap", a, b, cap * 1.1, 1.2)
        elif b.metabolism.p_repair > a.metabolism.p_repair + 8 and a.damage_X > b.damage_X:
            move_resource("p_repair", "p_repair_cap", b, a, cap * 1.1, 1.2)

        # Detoxificación cooperativa: la célula menos cargada absorbe una fracción
        # pequeña del residuo de una vecina más intoxicada. No destruye masa.
        if a.metabolism.waste_fraction > b.metabolism.waste_fraction + 0.18 and b.metabolism.waste_fraction < 0.55:
            actual = min(a.metabolism.w_waste, cap * 0.9,
                         max(0.0, b.metabolism.w_waste_cap - b.metabolism.w_waste))
            a.metabolism.w_waste -= actual
            b.metabolism.w_waste += actual
            b.provided_support += actual * 0.6
            a.received_support += actual * 0.6
            j.last_activity = self.tick_count
        elif b.metabolism.waste_fraction > a.metabolism.waste_fraction + 0.18 and a.metabolism.waste_fraction < 0.55:
            actual = min(b.metabolism.w_waste, cap * 0.9,
                         max(0.0, a.metabolism.w_waste_cap - a.metabolism.w_waste))
            b.metabolism.w_waste -= actual
            a.metabolism.w_waste += actual
            a.provided_support += actual * 0.6
            b.received_support += actual * 0.6
            j.last_activity = self.tick_count

    def _update_attachment_strengths(self):
        for c in self.cells.values():
            strengths = [
                self.junctions[jid].strength for jid in c.junction_ids
                if jid in self.junctions and self.junctions[jid].kind == JunctionKind.ADHESION
            ]
            c.attachment_strength = float(np.clip(sum(strengths) / 3.0, 0.0, 1.0))

    def _deliver_synaptic_events(self):
        for j in list(self.junctions.values()):
            if j.kind != JunctionKind.SYNAPTIC:
                continue
            post = self.cells.get(j.cell_b)
            if post is None or not post.alive:
                self._remove_junction(j.id)
                continue
            pending: List[Tuple[int, float]] = []
            for deliver_tick, amount in j.event_queue:
                if deliver_tick <= self.tick_count:
                    sign = -1.0 if j.receptor_type == "inhibitory" else 1.0
                    current = sign * amount * j.weight * j.signal_conductance
                    post.synaptic_input += np.array([current, current * 0.45, current * 0.25, -current * 0.15])
                    j.last_activity = self.tick_count
                else:
                    pending.append((deliver_tick, amount))
            j.event_queue = pending

    def _queue_synaptic_events(self):
        for j in list(self.junctions.values()):
            if j.kind != JunctionKind.SYNAPTIC:
                continue
            pre = self.cells.get(j.cell_a); post = self.cells.get(j.cell_b)
            if pre is None or post is None or not pre.alive or not post.alive:
                self._remove_junction(j.id)
                continue

            if pre.spike_output > 0.05:
                pre.last_spike_tick = self.tick_count
                j.last_pre_tick = self.tick_count
                j.event_queue.append((self.tick_count + j.delay, pre.spike_output))
                j.prune_score = max(0.0, j.prune_score - 0.08)

            if post.spike_output > 0.05:
                post.last_spike_tick = self.tick_count
                j.last_post_tick = self.tick_count

            if j.last_pre_tick is not None and j.last_post_tick is not None:
                dt = j.last_post_tick - j.last_pre_tick
                if -8 <= dt <= 8:
                    if dt >= 0:
                        j.weight += 0.012 * math.exp(-dt / 4.0)
                    else:
                        j.weight -= 0.010 * math.exp(dt / 4.0)
                    j.weight = float(np.clip(j.weight, -1.5, 1.8))
                    j.strength = float(np.clip(j.strength + 0.003 * abs(j.weight), 0.0, 1.0))
                    j.last_activity = self.tick_count

    def _differentiate_cells(self):
        if self.tick_count % 5 != 0:
            return
        alive_cells = [c for c in self.cells.values() if c.alive]
        type_counts: Dict[CellType, int] = {}
        for c in alive_cells:
            type_counts[c.cell_type] = type_counts.get(c.cell_type, 0) + 1
        role_floor = max(1, int(len(alive_cells) * 0.06))

        def scarcity(t: CellType) -> float:
            return max(0.0, (role_floor - type_counts.get(t, 0)) / max(1.0, role_floor))

        for c in alive_cells:
            if not c.alive:
                continue
            neighbors = self.world.get_neighbor_cells(c.x, c.y, 2, self.cells)
            neighbor_count = len(neighbors)
            neighbor_damage = np.mean([n["damage"] for n in neighbors]) if neighbors else 0.0
            adhesion_count = sum(1 for jid in c.junction_ids
                                 if jid in self.junctions and self.junctions[jid].kind == JunctionKind.ADHESION)
            syn_count = sum(1 for jid in c.junction_ids
                            if jid in self.junctions and self.junctions[jid].kind == JunctionKind.SYNAPTIC)
            local_signals = self.world.sample_signals(c.x, c.y)

            # Información posicional morfogenética: sesga diferenciación según
            # la posición de la célula en el gradiente heredable del genoma.
            morph_a, morph_b = self.world.sample_morphogens(c.x, c.y)
            morph_resp = c.genome.morphogen_response()   # (2, 4)
            # morph_bias[i]: BOUNDARY=0, NEURON=1, METABOLIC=2, REPAIR=3
            morph_bias = np.clip(
                morph_a * morph_resp[0] + morph_b * morph_resp[1], -0.30, 0.30
            )

            scores = {
                CellType.STEM: 0.15 if c.age_ticks < c.genome.development_ticks * 2 else 0.02,
                CellType.BOUNDARY: (0.25 if adhesion_count > 0 else 0.0) +
                                   max(0.0, 4 - neighbor_count) * 0.06 +
                                   c.boundary.c_integrity * 0.10 + scarcity(CellType.BOUNDARY) * 0.20 +
                                   float(morph_bias[0]),
                CellType.METABOLIC: c.metabolism.atp_fraction * 0.45 + min(1.0, c.metabolism.r_raw / 80.0) * 0.25 +
                                    scarcity(CellType.METABOLIC) * 0.52 + float(morph_bias[2]),
                CellType.REPAIR: neighbor_damage * 0.75 + c.homeostasis.g_damage_error * 0.35 +
                                 scarcity(CellType.REPAIR) * 0.50 + float(morph_bias[3]),
                CellType.SIGNALING: c.communication.z_signal_load * 0.80 + local_signals[SIGNAL_IDX["crowding"]] / 100.0 +
                                    scarcity(CellType.SIGNALING) * 0.46,
                CellType.NEURON: c.neural.t_excitation * 0.65 + syn_count * 0.18 + c.memory.h_integrity * 0.10 +
                                 scarcity(CellType.NEURON) * 0.20 + float(morph_bias[1]),
                CellType.SENSORY: c.communication.z_coherence * 0.18 + np.mean(local_signals > 3.0) * 0.28,
                CellType.MOTOR: c.genome.motility * 0.35 + abs(c.neural.t_action_bias[0]) * 0.20 + abs(c.neural.t_action_bias[1]) * 0.20,
                CellType.GERMLINE: c.reproduction.r_maturity * 0.65 + c.identity.I * 0.20 - c.damage_X * 0.4,
                CellType.POLICING: c.cheater_score * 0.15 + max(0.0, self.organism.shared_stress - 0.35) * 0.65,
            }
            target = max(scores, key=scores.get)
            current_score = scores.get(c.cell_type, 0.0)
            if target != c.cell_type and scores[target] > current_score + 0.12 and c.type_commitment < 0.82:
                cost = 0.18 + 0.25 * c.type_commitment
                if c.metabolism.consume_atp(cost) >= cost * 0.7:
                    old = c.cell_type
                    c.cell_type = target
                    c.type_commitment = min(1.0, c.type_commitment + 0.22)
                    c._log(f"DIFFERENTIATED:{old.value}->{target.value}")
            else:
                c.type_commitment = min(1.0, c.type_commitment + 0.025)

    def _police_cells(self):
        for c in self.cells.values():
            c.received_support *= 0.94
            c.provided_support *= 0.94
            c.contribution_score *= 0.995
            imbalance = c.received_support - c.provided_support * 1.8
            if imbalance > 1.0 and c.metabolism.atp_fraction > 0.38 and len(c.junction_ids) >= 2:
                c.cheater_score = min(1.0, c.cheater_score + 0.035 * imbalance)
            else:
                c.cheater_score = max(0.0, c.cheater_score - 0.015)

            if c.cheater_score > 0.72:
                c.reproduction.r_maturity = max(0.0, c.reproduction.r_maturity - 0.015)
                for jid in list(c.junction_ids):
                    j = self.junctions.get(jid)
                    if j:
                        j.strength = max(0.0, j.strength - 0.012)
                if c.cell_type == CellType.POLICING:
                    c.cheater_score *= 0.8

    def _apply_organism_pressure(self):
        if self.organism.collective_identity <= 0.0:
            return
        for c in self.cells.values():
            if not c.alive:
                continue
            if self.organism.shared_stress > 0.38:
                c.homeostasis.p_repair = min(0.85, c.homeostasis.p_repair + 0.015)
                c.homeostasis.p_reproduction = max(0.02, c.homeostasis.p_reproduction - 0.010)
            if self.organism.collective_identity > 0.52 and c.cell_type in (CellType.BOUNDARY, CellType.REPAIR):
                c.contribution_score = min(10.0, c.contribution_score + 0.002)

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
        if self.organism.collective_identity < 0.65 or self.organism.role_coverage < 0.8:
            return
        if self.organism.reproduction_pressure < 0.25:
            return

        germline = [
            c for c in self.cells.values()
            if c.alive and c.cell_type == CellType.GERMLINE and
            c.reproduction.r_maturity > 0.7 and c.identity.I > 0.55 and
            c.metabolism.a_free > 20 and c.metabolism.m_struct > 25
        ]
        if not germline:
            return
        parent = max(germline, key=lambda c: c.identity.I + c.metabolism.atp_fraction)
        positions = self._free_positions_near(parent.x, parent.y, 2, 3)
        if len(positions) < 3:
            return

        child_genome = parent.reproduction.build_offspring_genome(
            parent.genome, parent.neural, parent.memory, parent.rng
        )
        if child_genome is None:
            parent._log("ORGANISM_REPRO_ABORTED:heredity")
            return

        atp_cost = 14.0
        struct_cost = 10.0
        if parent.metabolism.consume_atp(atp_cost) < atp_cost * 0.8:
            return
        if parent.metabolism.consume_structural(struct_cost) < struct_cost * 0.8:
            return

        seed_types = [CellType.STEM, CellType.BOUNDARY, CellType.METABOLIC]
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

        if len(created) >= 2:
            self._create_junction(JunctionKind.ADHESION, created[0].id, created[1].id, 0.55, 0.04, 0.10)
        if len(created) >= 3:
            self._create_junction(JunctionKind.ADHESION, created[1].id, created[2].id, 0.55, 0.04, 0.10)
            self._create_junction(JunctionKind.METABOLIC, created[0].id, created[2].id, 0.42, 0.24, 0.12)
        parent.reproduction.r_maturity = max(0.0, parent.reproduction.r_maturity - 0.35)
        parent._log("ORGANISM_REPRODUCTION:seed_cluster")
        self.organism.update(self.cells, self.junctions)

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

        avg_I = sum(c.identity.I for c in alive) / (len(alive) + 1e-9)

        return {
            "tick": self.tick_count,
            "alive": len(alive),
            "max": self.max_cells,
            "phases": phases,
            "generations": gen_dist,
            "cell_types": type_dist,
            "avg_identity_I": round(avg_I, 3),
            "organism": self.organism.to_dict(),
            "junction_count": len(self.junctions),
            "junctions": [j.to_dict() for j in list(self.junctions.values())[:80]],
            "recent_deaths": self.dead_log[-5:],
            "cells": [c.get_status() for c in alive]
        }


# ─────────────────────────────────────────────────────────────
# MOTOR DE EVOLUCIÓN — selección por torneo entre colonias
# ─────────────────────────────────────────────────────────────

class EvolutionEngine:
    """
    Ejecuta N_COLONIES colonias aisladas en paralelo (round-robin secuencial).
    Cada TOURNAMENT_INTERVAL ticks: clasifica colonias por fitness, mata la mitad
    inferior y la reemplaza clonando genomas ganadores con mutación adicional.

    Esto es selección por torneo multinivel: las células ya evolucionan dentro
    de cada colonia; el motor añade selección entre colonias, haciendo que los
    genomas más aptos dominen la población a lo largo de generaciones.
    """

    def __init__(self, n_colonies: int, tournament_interval: int, rng: random.Random):
        self.n_colonies = n_colonies
        self.tournament_interval = tournament_interval
        self.rng = rng
        self.tick_count = 0
        self.evolution_history: List[Dict] = []

        # Genoma primordial compartido; cada colonia arranca de una variante mutada
        primordial = Genome.create(rng)

        self.worlds: List[SpatialWorld] = []
        self.colonies: List[Colony] = []
        self.founding_genomes: List[Genome] = []
        self._cached_best_idx: int = 0
        self._cached_fitnesses: List[float] = []

        per_colony_cells = max(4, MAX_CELLS // n_colonies)
        for i in range(n_colonies):
            scale = derive_simulation_scale(per_colony_cells)
            col_rng = random.Random(rng.randint(0, 2**31))
            world = SpatialWorld(
                width=scale.world_width, height=scale.world_height,
                n_sources=scale.n_sources, rng=col_rng, source_strength=8.0
            )
            for _ in range(scale.prewarm_ticks):
                world.tick()
            colony = Colony(world, max_cells=scale.max_cells)
            colony.rng = col_rng
            founding = primordial.mutate(rng)
            colony.spawn_primordial_from_genome(scale.primordial_cells, founding)
            self.worlds.append(world)
            self.colonies.append(colony)
            self.founding_genomes.append(founding)

    # ── Fitness de una colonia ─────────────────────────────────

    def _colony_fitness(self, colony: Colony) -> float:
        alive = [c for c in colony.cells.values() if c.alive]
        if not alive:
            return 0.0
        # +1 so generation-0 cells still produce positive fitness
        mean_gen = 1.0 + sum(c._generation for c in alive) / len(alive)
        mean_I   = sum(c.identity.I  for c in alive) / len(alive)
        stage_mult = {
            "solitary":    0.5,
            "aggregate":   0.8,
            "proto_tissue":1.2,
            "integrated":  2.0,
            "extinct":     0.0,
        }.get(colony.organism.development_stage, 0.5)
        return mean_gen * mean_I * stage_mult

    # ── Torneo de selección ────────────────────────────────────

    def _reset_colony(self, idx: int, founding_genome: Genome):
        """Destruye y recrea la colonia idx con el nuevo genoma fundador."""
        old_colony = self.colonies[idx]
        old_world  = self.worlds[idx]

        # Vaciar celulas y uniones
        old_colony.cells.clear()
        old_colony.junctions.clear()
        old_colony._junction_pairs.clear()
        old_colony.organism = OrganismState()
        old_colony.tick_count = 0
        old_colony.dead_log.clear()

        # Reiniciar campos del mundo
        old_world.nutrients[:] = 0.0
        old_world.toxins[:]    = 0.0
        old_world.signals[:]   = 0.0
        old_world.occupied.clear()
        old_world.tick_count   = 0
        # Re-sembrar fuentes y pre-calentar
        for sx, sy in old_world.sources:
            old_world.nutrients[sy, sx] = 80.0
        scale = derive_simulation_scale(old_colony.max_cells)
        for _ in range(scale.prewarm_ticks):
            old_world.tick()

        self.founding_genomes[idx] = founding_genome
        old_colony.spawn_primordial_from_genome(scale.primordial_cells, founding_genome)

    def _run_tournament(self):
        fitnesses = [self._colony_fitness(c) for c in self.colonies]
        ranked = sorted(range(self.n_colonies), key=lambda i: fitnesses[i], reverse=True)
        n_winners = self.n_colonies // 2
        winners = ranked[:n_winners]
        losers  = ranked[n_winners:]

        # Snapshot del mejor genoma para el registro histórico
        best_idx = winners[0]
        best_colony = self.colonies[best_idx]
        best_alive = [c for c in best_colony.cells.values() if c.alive]
        best_genome_snap: Optional[Dict] = None
        if best_alive:
            best_cell = max(best_alive, key=lambda c: c.identity.I * c.genome.fidelity)
            best_genome_snap = {
                "fidelity": round(best_cell.genome.fidelity, 3),
                "membrane_strength": round(best_cell.genome.membrane_strength, 3),
                "metabolic_base_rate": round(best_cell.genome.metabolic_base_rate, 3),
                "neural_plasticity": round(best_cell.genome.neural_plasticity, 4),
                "motility": round(best_cell.genome.motility, 3),
            }

        for rank, li in enumerate(losers):
            wi = winners[rank % len(winners)]
            winner_colony = self.colonies[wi]
            winner_alive  = [c for c in winner_colony.cells.values() if c.alive]
            if winner_alive:
                donor = max(winner_alive,
                            key=lambda c: c.identity.I * c.genome.fidelity)
                new_genome = donor.genome.mutate(self.rng)
            else:
                new_genome = self.founding_genomes[wi].mutate(self.rng)
            self._reset_colony(li, new_genome)

        self.evolution_history.append({
            "tournament": len(self.evolution_history) + 1,
            "tick": self.tick_count,
            "fitnesses": [round(f, 4) for f in fitnesses],
            "winner_indices": winners,
            "loser_indices": losers,
            "best_fitness": round(fitnesses[best_idx], 4),
            "mean_fitness": round(sum(fitnesses) / len(fitnesses), 4),
            "best_genome": best_genome_snap,
        })
        if len(self.evolution_history) > 20:
            self.evolution_history.pop(0)

    # ── Tick principal ─────────────────────────────────────────

    def tick(self):
        for colony in self.colonies:
            colony.tick(include_status=False)
        self.tick_count += 1
        self._cached_fitnesses = [self._colony_fitness(c) for c in self.colonies]
        if self._cached_fitnesses:
            self._cached_best_idx = int(np.argmax(self._cached_fitnesses))
        if self.tick_count % self.tournament_interval == 0:
            self._run_tournament()

    # ── Estado para HTTP ───────────────────────────────────────

    def best_colony_idx(self) -> int:
        return self._cached_best_idx

    def status(self) -> Dict:
        fitnesses = self._cached_fitnesses or [self._colony_fitness(c) for c in self.colonies]
        best_idx  = self._cached_best_idx
        return {
            "tick": self.tick_count,
            "n_colonies": self.n_colonies,
            "tournament_interval": self.tournament_interval,
            "tournaments_run": len(self.evolution_history),
            "colony_fitnesses": [round(f, 4) for f in fitnesses],
            "best_colony_idx": best_idx,
            "colony_sizes": [
                sum(1 for c in col.cells.values() if c.alive)
                for col in self.colonies
            ],
            "colony_stages": [
                col.organism.development_stage for col in self.colonies
            ],
            "founding_genome_fidelities": [
                round(g.fidelity, 3) for g in self.founding_genomes
            ],
            "evolution_history": self.evolution_history[-5:],
        }


# ─────────────────────────────────────────────────────────────
# HTTP SERVER / DASHBOARD
# ─────────────────────────────────────────────────────────────

EVOLUTION_ENGINE: Optional["EvolutionEngine"] = None

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>VIDA DIGITAL v6 — CNDV</title>
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
  <h1>VIDA DIGITAL v6</h1>
  <div class="subtitle">CNDV — 9 bloques | B1-B9 | M1-M3 | H1-H4 | D1-D7</div>
  <span id="tick-counter">tick 0</span>
  <div id="speed-control">
    vel: <button onclick="setSpeed(50)">1×</button>
         <button onclick="setSpeed(20)">2×</button>
         <button onclick="setSpeed(5)">5×</button>
  </div>
</div>
<div id="main">
  <!-- Panel izquierdo: Colonia -->
  <div class="panel" id="left-panel">
    <div class="section-title">Colonia</div>
    <div id="colony-stats"></div>
    <div class="section-title">Evolución — Torneo</div>
    <div id="evo-stats"></div>
    <div id="evo-fitness-bars"></div>
    <div class="section-title">Células vivas</div>
    <div id="cells-list"></div>
    <div class="section-title">Muertes recientes</div>
    <div id="deaths-log"></div>
  </div>

  <!-- Panel central: Mundo + gráfica -->
  <div id="world-panel">
    <canvas id="world-canvas" width="400" height="400"></canvas>
    <div style="width:400px; margin-top:8px;">
      <div class="section-title">Identidad I — colonia</div>
      <canvas id="identity-chart" width="400" height="80"></canvas>
    </div>
  </div>

  <!-- Panel derecho: Célula seleccionada -->
  <div class="panel" id="right-panel">
    <div class="section-title">Célula seleccionada</div>
    <div id="cell-detail">
      <div style="color:#4a7090; padding: 20px 0; text-align:center;">Selecciona una célula</div>
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
    <div class="colony-stat"><div class="val">${s.alive}</div><div class="lbl">vivas</div></div>
    <div class="colony-stat"><div class="val">${(s.avg_identity_I*100).toFixed(0)}%</div><div class="lbl">I media</div></div>
    <div class="colony-stat"><div class="val">${s.junction_count||0}</div><div class="lbl">junctions</div></div>
    <div class="colony-stat"><div class="val">${((org.collective_identity||0)*100).toFixed(0)}%</div><div class="lbl">I organismo</div></div>
    <div class="colony-stat"><div class="val">${Object.keys(s.cell_types||{}).length}</div><div class="lbl">tipos</div></div>
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
      <div style="color:#4a7090;font-size:10px">tipo: ${c.cell_type || 'n/a'}</div>
      ${barRow('I identidad', 'bar-identity', S.I)}
      ${barRow('ATP', 'bar-atp', S.A)}
      ${barRow('Frontera', 'bar-boundary', S.C)}
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
    {lbl:'M→A', val: S.A, tip:'metabolismo→ATP'},
    {lbl:'A→R', val: 1-(S.X||0), tip:'ATP→reparación'},
    {lbl:'R→B1', val: S.C, tip:'rep→frontera'},
    {lbl:'B1→M', val: S.C*(S.C||0), tip:'frontera→metabolismo'},
    {lbl:'Cog→', val: B4.excitation||0, tip:'cogn→acción'}
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
      <span class="gen-badge" style="margin-left:5px">Generación ${c.generation}</span>
      <div style="color:#4a7090;font-size:10px">edad ${c.age} ticks | pos (${c.position.x},${c.position.y})</div>
    </div>

    <div class="section-title">B9 Identidad — I = ${((S.I||0)*100).toFixed(0)}%</div>
    ${barRow('I total', 'bar-identity', S.I)}
    <div style="font-size:10px;color:#4a7090;margin:4px 0">Clausura organizacional:</div>
    <div class="closure-ring">${closureHtml}</div>

    <div class="section-title">B1 Frontera</div>
    ${barRow('integridad', 'bar-boundary', S.C)}
    ${barRow('daño permanente', 'bar-damage', c.B1_boundary?.c_permanent_damage)}

    <div class="section-title">B2 Metabolismo</div>
    ${barRow('ATP', 'bar-atp', S.A)}
    ${barRow('residuo W', 'bar-damage', S.W)}
    ${barRow('eficiencia η', 'bar-neural', c.B2_metabolism?.eta_metabolic)}
    <div class="stat-row"><span class="stat-label">masa_struct</span><span class="stat-value">${(S.M||0).toFixed(1)}</span></div>
    <div class="stat-row"><span class="stat-label">mat_reparativo</span><span class="stat-value">${(S.P||0).toFixed(1)}</span></div>

    <div class="section-title">B3 Homeostasis (red regulatoria)</div>
    <div style="font-size:10px;color:#4a7090;margin:2px 0">Prioridades ← W_reg @ errores:</div>
    <div class="priorities-bar">
      <div class="pri-maint" style="width:${((pri.maintenance||0)*100).toFixed(0)}%" title="mantenimiento"></div>
      <div class="pri-repair" style="width:${((pri.repair||0)*100).toFixed(0)}%" title="reparación"></div>
      <div class="pri-action" style="width:${((pri.action||0)*100).toFixed(0)}%" title="acción"></div>
      <div class="pri-repro" style="width:${((pri.reproduction||0)*100).toFixed(0)}%" title="reproducción"></div>
    </div>
    <div style="display:flex;gap:8px;font-size:9px;margin:2px 0">
      <span style="color:#40c8ff">■ maint ${((pri.maintenance||0)*100).toFixed(0)}%</span>
      <span style="color:#40c080">■ rep ${((pri.repair||0)*100).toFixed(0)}%</span>
      <span style="color:#ffa040">■ act ${((pri.action||0)*100).toFixed(0)}%</span>
      <span style="color:#ff40ff">■ repr ${((pri.reproduction||0)*100).toFixed(0)}%</span>
    </div>
    ${barRow('estrés sistémico', 'bar-stress', B3.stress)}

    <div class="section-title">B4 Sustrato neuronal</div>
    ${barRow('excitación', 'bar-neural', B4.excitation)}
    ${barRow('integridad', 'bar-boundary', B4.integrity)}
    ${barRow('error predictivo', 'bar-stress', B4.prediction_error)}
    <div class="stat-row"><span class="stat-label">action_bias</span>
      <span class="stat-value">[${(B4.action_bias||[0,0,0]).map(v=>v.toFixed(2)).join(', ')}]</span></div>

    <div class="section-title">B5 Memoria material</div>
    ${barRow('integridad', 'bar-mem', B5.h_integrity)}
    ${barRow('tasa corrupción', 'bar-damage', (B5.corruption_rate||0)*20)}
    <div class="stat-row"><span class="stat-label">h_regulatory</span>
      <span class="stat-value">[${(B5.h_regulatory||[]).map(v=>v.toFixed(2)).join(',')}]</span></div>

    <div class="section-title">B7 Reproducción</div>
    ${barRow('madurez', 'bar-atp', B7.maturity)}
    ${barRow('riesgo fallo', 'bar-damage', B7.failure_risk)}
    <div class="stat-row"><span class="stat-label">replicando</span>
      <span class="stat-value" style="color:${B7.replicating?'#ffa040':'#4a7090'}">${B7.replicating?'SÍ (etapa '+B7.d_stage+')':'no'}</span></div>

    <div class="section-title">B8 Comunicación biosemiótica</div>
    ${barRow('carga señal', 'bar-neural', B8.signal_load || 0)}
    ${barRow('coherencia', 'bar-mem', B8.coherence || 0)}
    <div class="stat-row"><span class="stat-label">mov social</span>
      <span class="stat-value">[${(B8.social_move||[0,0]).map(v=>v.toFixed(2)).join(', ')}]</span></div>
    <div class="stat-row"><span class="stat-label">recibido</span>
      <span class="stat-value">${Object.entries(B8.received||{}).filter(([k,v])=>v>0.02).map(([k,v])=>k+':'+v.toFixed(2)).join(' | ') || 'silencio'}</span></div>

    <div class="section-title">Capa multicelular</div>
    ${barRow('adhesión', 'bar-boundary', MC.attachment_strength || 0)}
    ${barRow('spike', 'bar-neural', MC.spike_output || 0)}
    ${barRow('cheater', 'bar-damage', MC.cheater_score || 0)}
    <div class="stat-row"><span class="stat-label">junctions</span><span class="stat-value">${MC.junction_count || 0}</span></div>
    <div class="stat-row"><span class="stat-label">soporte</span>
      <span class="stat-value">+${(MC.provided_support||0).toFixed(2)} / -${(MC.received_support||0).toFixed(2)}</span></div>

    <div class="section-title">Eventos recientes</div>
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
    document.getElementById('evo-stats').innerHTML = '<div style="color:#4a7090;padding:4px 0">inicializando…</div>';
    document.getElementById('evo-fitness-bars').innerHTML = '';
    return;
  }
  const fitnesses = evo.colony_fitnesses || [];
  const best_idx  = evo.best_colony_idx || 0;
  const maxFit    = Math.max(...fitnesses, 0.001);
  const meanFit   = fitnesses.reduce((a, b) => a + b, 0) / Math.max(1, fitnesses.length);
  document.getElementById('evo-stats').innerHTML = `
    <div class="stat-row"><span class="stat-label">torneos</span><span class="stat-value">${evo.tournaments_run}</span></div>
    <div class="stat-row"><span class="stat-label">colonias</span><span class="stat-value">${evo.n_colonies}</span></div>
    <div class="stat-row"><span class="stat-label">mejor fit</span><span class="stat-value">${(fitnesses[best_idx]||0).toFixed(3)}</span></div>
    <div class="stat-row"><span class="stat-label">media fit</span><span class="stat-value">${meanFit.toFixed(3)}</span></div>
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

    // actualizar celula seleccionada si sigue viva
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
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            if EVOLUTION_ENGINE:
                best = EVOLUTION_ENGINE.best_colony_idx()
                body = json.dumps(EVOLUTION_ENGINE.colonies[best].status())
            else:
                body = '{"tick":0,"alive":0,"cells":[],"loading":true}'
            self.wfile.write(body.encode())
        elif self.path == '/world':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            if EVOLUTION_ENGINE:
                best = EVOLUTION_ENGINE.best_colony_idx()
                body = json.dumps(EVOLUTION_ENGINE.worlds[best].snapshot())
            else:
                body = '{"cells":[],"loading":true}'
            self.wfile.write(body.encode())
        elif self.path == '/evolution':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
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


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    global EVOLUTION_ENGINE, MAX_CELLS, N_COLONIES, TOURNAMENT_INTERVAL

    ap = argparse.ArgumentParser(description="VIDA DIGITAL v6 — CNDV")
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
    args = ap.parse_args()

    MAX_CELLS           = args.max_cells
    N_COLONIES          = args.n_colonies
    TOURNAMENT_INTERVAL = args.tournament_interval

    print("=" * 60)
    print("  VIDA DIGITAL v6 — CNDV + EvolutionEngine")
    print("  B1-B9 | M1-M3 | H1-H4 | D1-D7 | Torneo multinivel")
    print(f"  {N_COLONIES} colonias | torneo cada {TOURNAMENT_INTERVAL} ticks | max_cells={MAX_CELLS:,}")
    print("=" * 60)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    server = run_server(args.port)
    print(f"\n  Dashboard → http://localhost:{args.port}")
    print(f"  Estado    → http://localhost:{args.port}/status")
    print(f"  Mundo     → http://localhost:{args.port}/world")
    print(f"  Evolución → http://localhost:{args.port}/evolution")
    print(f"\n  Inicializando {N_COLONIES} colonias...")

    EVOLUTION_ENGINE = EvolutionEngine(
        n_colonies=N_COLONIES,
        tournament_interval=TOURNAMENT_INTERVAL,
        rng=rng,
    )

    per_colony_cells = max(4, MAX_CELLS // N_COLONIES)
    scale = derive_simulation_scale(per_colony_cells)
    print(f"  Escala por colonia → max_cells={scale.max_cells:,} | "
          f"mundo={scale.world_width}×{scale.world_height} | "
          f"fuentes={scale.n_sources}")
    print(f"\n  Iniciando simulación...\n")

    tick = 0
    try:
        while True:
            EVOLUTION_ENGINE.tick()
            tick += 1

            if tick % 100 == 0:
                fitnesses = [EVOLUTION_ENGINE._colony_fitness(c)
                             for c in EVOLUTION_ENGINE.colonies]
                best_idx = int(np.argmax(fitnesses))
                best_col = EVOLUTION_ENGINE.colonies[best_idx]
                alive    = sum(1 for c in best_col.cells.values() if c.alive)
                avg_I    = sum(c.identity.I for c in best_col.cells.values() if c.alive)
                avg_I   /= max(1, alive)
                gens     = set(c._generation for c in best_col.cells.values() if c.alive)
                n_tournaments = len(EVOLUTION_ENGINE.evolution_history)
                print(
                    f"t={tick:6d} | best_col={best_idx} "
                    f"fit={fitnesses[best_idx]:.3f} "
                    f"mean_fit={sum(fitnesses)/len(fitnesses):.3f} "
                    f"| vivas={alive:3d} I={avg_I:.2f} "
                    f"gen={max(gens) if gens else 0} "
                    f"stage={best_col.organism.development_stage} "
                    f"| torneos={n_tournaments}"
                )

                # Reseed extintas inmediatamente (no esperar el torneo)
                for i, col in enumerate(EVOLUTION_ENGINE.colonies):
                    if not any(c.alive for c in col.cells.values()):
                        fg = EVOLUTION_ENGINE.founding_genomes[i]
                        col.spawn_primordial_from_genome(3, fg)

            time.sleep(0.04)

    except KeyboardInterrupt:
        n_t = len(EVOLUTION_ENGINE.evolution_history)
        print(f"\n\n  Simulación detenida. Ticks={tick} | Torneos={n_t}")
        if EVOLUTION_ENGINE.evolution_history:
            last = EVOLUTION_ENGINE.evolution_history[-1]
            print(f"  Último torneo: mejor_fitness={last['best_fitness']:.3f} "
                  f"media={last['mean_fitness']:.3f}")
        server.shutdown()


if __name__ == '__main__':
    main()
