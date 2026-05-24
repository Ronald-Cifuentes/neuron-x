import importlib.util
import json
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_nx1():
    spec = importlib.util.spec_from_file_location("nx1", ROOT / "nx-1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_colony(nx1, *, max_cells=32, seed=1):
    rng = random.Random(seed)
    np.random.seed(seed)
    world = nx1.SpatialWorld(width=20, height=20, n_sources=2, rng=rng)
    colony = nx1.Colony(world, max_cells=max_cells)
    colony.rng = rng
    return colony, rng


def add_cell(nx1, colony, rng, x, y, cell_type, *, identity=0.9):
    cell = nx1.Cell(
        x,
        y,
        colony.world,
        nx1.Genome.create(rng),
        rng=random.Random(rng.randint(0, 2**31)),
    )
    cell.cell_type = cell_type
    cell.identity.I = identity
    cell.homeostasis.g_stress = 0.0
    cell.damage_X = 0.0
    cell.boundary.c_integrity = 0.95
    cell.metabolism.a_free = 140.0
    cell.metabolism.m_struct = 90.0
    cell.metabolism.r_raw = 80.0
    cell.metabolism.p_repair = 40.0
    colony.cells[cell.id] = cell
    return cell


def test_simulation_scale_covers_requested_cell_budget():
    nx1 = load_nx1()

    scale = nx1.derive_simulation_scale(97)

    assert scale.max_cells == 97
    assert scale.world_capacity >= 97
    assert scale.primordial_cells <= 97
    assert scale.n_sources > 0


def test_spatial_world_clamps_diffuses_and_reports_local_fields():
    nx1 = load_nx1()
    rng = random.Random(2)
    world = nx1.SpatialWorld(width=12, height=12, n_sources=1, rng=rng, source_strength=0.0)

    world.deposit_nutrient(3, 3, 500.0)
    world.deposit_toxin(3, 3, 500.0)
    world.deposit_signal(3, 3, "repair_need", 500.0)
    consumed = world.consume_nutrient(3, 3, 20.0)
    before_signal = world.sample_signals(3, 3)[nx1.SIGNAL_IDX["repair_need"]]

    world.tick()
    nutrient, toxin = world.sample(3, 3)
    after_signal = world.sample_signals(3, 3)[nx1.SIGNAL_IDX["repair_need"]]
    morph_a, morph_b = world.sample_morphogens(0, 0)

    assert consumed == 20.0
    assert 0.0 <= nutrient <= 120.0
    assert 0.0 <= toxin <= 60.0
    assert 0.0 <= after_signal <= before_signal
    assert morph_a == pytest.approx(1.0)
    assert morph_b == pytest.approx(1.0)


def test_genome_create_mutate_preserves_required_shapes_and_ranges():
    nx1 = load_nx1()
    rng = random.Random(3)
    genome = nx1.Genome.create(rng)
    mutated = genome.mutate(rng)

    assert genome.W_reg().shape == (4, 5)
    assert genome.neural_W_ih().shape == (4, 6)
    assert genome.neural_W_hh().shape == (4, 4)
    assert genome.neural_W_ho().shape == (3, 4)
    assert genome.signal_receptor_W().shape == (4, nx1.N_SIGNAL_CHANNELS)
    assert genome.morphogen_response().shape == (2, 4)
    assert 0.1 <= mutated.membrane_strength <= 1.0
    assert 0.1 <= mutated.conversion_efficiency <= 0.85
    assert 0.001 <= mutated.neural_plasticity <= 0.1
    assert 1 <= mutated.sensor_range <= 6


def test_boundary_degrades_repairs_and_neural_gate_changes_permeability():
    nx1 = load_nx1()
    genome = nx1.Genome()
    boundary = nx1.Boundary(c_integrity=0.8, c_permeability_resource=0.7)

    damage = boundary.degrade(tox_external=30.0, tox_internal=20.0, age_ticks=100, genome=genome)
    damaged_integrity = boundary.c_integrity
    atp_used, mass_used = boundary.repair(50.0, 50.0, 1.0, genome)
    before_gate = boundary.c_permeability_resource
    boundary.apply_neural_gate(0.1)

    assert damage > 0.0
    assert damaged_integrity < 0.8
    assert atp_used > 0.0
    assert mass_used > 0.0
    assert boundary.c_integrity > damaged_integrity
    assert boundary.c_neural_gate == pytest.approx(0.2)
    assert boundary.c_permeability_resource <= before_gate


def test_metabolism_conserves_caps_produces_waste_and_repairs_efficiency():
    nx1 = load_nx1()
    genome = nx1.Genome()
    met = nx1.Metabolism(r_raw=80.0, a_free=80.0, w_waste=10.0)

    out = met.step(genome, tox_internal=10.0, maintenance_priority=1.0)
    waste_before = met.w_waste
    eta_before = met.eta_metabolic
    cleared = met.repair_efficiency(5.0)

    assert out["atp_produced"] > 0.0
    assert out["waste_produced"] > 0.0
    assert met.r_raw < 80.0
    assert 0.0 <= met.a_free <= met.a_free_cap
    assert waste_before > 10.0
    assert cleared > 0.0
    assert met.w_waste < waste_before
    assert met.eta_metabolic >= eta_before
    assert met.consume_atp(10_000.0) <= met.a_free_cap
    assert met.a_free == 0.0


def test_homeostasis_computes_normalized_priorities_and_adapts_weights():
    nx1 = load_nx1()
    genome = nx1.Genome()
    homeostasis = nx1.HomeostasisRegulator(genome)
    met = nx1.Metabolism(a_free=5.0, w_waste=50.0)
    boundary = nx1.Boundary(c_integrity=0.2)

    out = homeostasis.update(met, boundary, damage=0.7, mem_integrity=0.2, mem_regulatory=np.zeros(5))
    priorities = out["priorities"]
    before = homeostasis.W_reg.copy()
    homeostasis.g_stress = 0.0
    homeostasis.adapt_W_reg(learning_signal=1.0)

    assert sum(priorities.values()) == pytest.approx(1.0)
    assert out["stress"] > 0.0
    assert out["errors"]["energy"] > 0.0
    assert out["errors"]["boundary"] > 0.0
    assert np.any(homeostasis.W_reg != before)


def test_neural_core_predicts_learns_degrades_and_repairs():
    nx1 = load_nx1()
    np.random.seed(4)
    genome = nx1.Genome()
    neural = nx1.NeuralCore(genome)
    inputs_a = np.array([0.8, 0.9, 0.1, 0.7, 0.0, 0.1])
    inputs_b = np.array([0.4, 0.6, 0.2, 0.1, 0.8, 0.3])

    neural.step(inputs_a, np.zeros(4), atp_available=20.0)
    weights_before = neural.W_ih.copy()
    out = neural.step(inputs_b, np.zeros(4), atp_available=20.0)
    neural.step(inputs_b, np.zeros(4), atp_available=1.0)
    noisy_integrity = neural.t_integrity
    neural.repair_weights(2.0)

    assert -1.0 <= out["move_x"] <= 1.0
    assert -1.0 <= out["move_y"] <= 1.0
    assert 0.0 <= out["capture_modulation"] <= 1.0
    assert neural.t_prediction_error > 0.0
    assert np.any(neural.W_ih != weights_before)
    assert neural.t_integrity >= noisy_integrity


def test_material_memory_updates_biases_degrades_and_repairs():
    nx1 = load_nx1()
    np.random.seed(5)
    memory = nx1.MaterialMemory(nx1.Genome())

    out = memory.update(
        boundary_integrity=0.4,
        homeostatic_errors={"energy": 0.4, "damage": 0.2, "waste": 0.1, "boundary": 0.5, "memory": 0.3},
        neural_trace=np.array([0.1, -0.2, 0.3, -0.4]),
        atp_available=10.0,
        waste_internal=30.0,
        damage=0.5,
    )
    integrity_after_update = memory.h_integrity
    memory.repair_memory(2.0)

    assert out["maintenance_cost"] > 0.0
    assert memory.get_regulatory_bias().shape == (5,)
    assert memory.get_adaptive_bias().shape == (4,)
    assert memory.get_heritable_snapshot().shape == (5,)
    assert memory.h_integrity >= integrity_after_update


def test_repair_system_repairs_boundary_memory_neural_and_metabolism():
    nx1 = load_nx1()
    genome = nx1.Genome()
    met = nx1.Metabolism(a_free=120.0, p_repair=50.0, w_waste=40.0, eta_metabolic=0.4)
    boundary = nx1.Boundary(c_integrity=0.4)
    neural = nx1.NeuralCore(genome)
    neural.t_noise_level = 0.8
    memory = nx1.MaterialMemory(genome)
    memory.h_integrity = 0.4
    homeostasis = nx1.HomeostasisRegulator(genome)
    homeostasis.p_repair = 1.0
    repair = nx1.RepairSystem(genome)

    first = repair.execute(met, boundary, neural, memory, homeostasis, damage=0.8, genome=genome)
    second = repair.execute(met, boundary, neural, memory, homeostasis, damage=0.8, genome=genome)

    assert first == (0.0, {})
    assert second[0] > 0.0
    assert second[1]["atp_used"] > 0.0
    assert boundary.c_integrity > 0.4
    assert neural.t_noise_level < 0.8
    assert memory.h_integrity > 0.4
    assert met.w_waste < 40.0


def test_reproduction_requires_all_gates_and_development_can_succeed_or_abort():
    nx1 = load_nx1()
    genome = nx1.Genome(development_ticks=2, repr_min_age=1)
    met = nx1.Metabolism(a_free=120.0, m_struct=80.0, q_repro=60.0)
    memory = nx1.MaterialMemory(genome)
    homeostasis = nx1.HomeostasisRegulator(genome)
    reproduction = nx1.ReproductionModule(genome)

    assert not reproduction.can_reproduce(met, nx1.Boundary(), memory, homeostasis, 0.0, 0.9, genome)
    reproduction.tick_maturity(age=1, genome=genome, met=met, damage=0.0)
    reproduction.r_maturity = 0.8
    reproduction.r_material = 20.0
    assert reproduction.can_reproduce(met, nx1.Boundary(), memory, homeostasis, 0.0, 0.9, genome)

    reproduction.initiate()
    reproduction.r_failure_risk = 0.0
    assert reproduction.develop_tick(met, genome) == "continue"
    assert reproduction.develop_tick(met, genome) == "success"

    starving = nx1.ReproductionModule(genome)
    starving.initiate()
    poor_met = nx1.Metabolism(a_free=0.0, m_struct=0.0)
    for _ in range(5):
        assert starving.develop_tick(poor_met, genome) == "continue"
    assert starving.develop_tick(poor_met, genome) == "abort"


def test_identity_combines_block_state_and_detects_sustained_organizational_death():
    nx1 = load_nx1()
    genome = nx1.Genome()
    identity = nx1.Identity()
    boundary = nx1.Boundary(c_integrity=0.9)
    met = nx1.Metabolism(a_free=100.0)
    met.atp_produced_last_tick = 5.0
    memory = nx1.MaterialMemory(genome)
    homeostasis = nx1.HomeostasisRegulator(genome)
    neural = nx1.NeuralCore(genome)
    neural.t_excitation = 0.5

    healthy = identity.compute(boundary, met, memory, homeostasis, neural, damage=0.0)
    identity._history = [0.01] * 10

    assert 0.0 <= healthy <= 1.0
    assert healthy > 0.7
    assert identity.is_organizationally_dead


def test_communication_emits_costly_signals_receives_and_modulates_homeostasis():
    nx1 = load_nx1()
    rng = random.Random(6)
    world = nx1.SpatialWorld(width=12, height=12, n_sources=1, rng=rng)
    genome = nx1.Genome()
    communication = nx1.CommunicationSystem(genome)
    met = nx1.Metabolism(a_free=80.0)
    boundary = nx1.Boundary(c_integrity=0.9)
    homeostasis = nx1.HomeostasisRegulator(genome)
    memory = nx1.MaterialMemory(genome)
    reproduction = nx1.ReproductionModule(genome)
    identity = nx1.Identity()
    identity.I = 0.9

    emitted = communication.emit(
        world, 4, 4, met, boundary, homeostasis, memory, reproduction, identity,
        damage=0.4, nut_local=80.0, tox_local=30.0, phase=nx1.LifePhase.STRESSED,
    )
    received = communication.receive(world, 4, 4, boundary, met, homeostasis, memory, damage=0.4)
    before = np.array([
        homeostasis.p_maintenance,
        homeostasis.p_repair,
        homeostasis.p_action,
        homeostasis.p_reproduction,
    ])
    communication.modulate_homeostasis(homeostasis)
    after = np.array([
        homeostasis.p_maintenance,
        homeostasis.p_repair,
        homeostasis.p_action,
        homeostasis.p_reproduction,
    ])

    assert emitted["cost"] > 0.0
    assert emitted["paid"] > 0.0
    assert np.any(world.sample_signals(4, 4) > 0.0)
    assert received["received"].shape == (nx1.N_SIGNAL_CHANNELS,)
    assert received["decoded"].shape == (4,)
    assert not np.allclose(before, after)
    assert after.sum() == pytest.approx(1.0)


def test_cell_tick_advances_integrated_lifecycle_and_status_contains_all_blocks():
    nx1 = load_nx1()
    colony, rng = make_colony(nx1, seed=66)
    cell = add_cell(nx1, colony, rng, 5, 5, nx1.CellType.METABOLIC)
    colony.world.deposit_nutrient(5, 5, 50.0)

    offspring = cell.tick()
    status = cell.get_status()

    assert offspring is None
    assert cell.age_ticks == 1
    assert cell.alive
    for key in (
        "B1_boundary",
        "B2_metabolism",
        "B3_homeostasis",
        "B4_neural",
        "B5_memory",
        "B6_repair",
        "B7_reproduction",
        "B8_communication",
        "B9_identity",
    ):
        assert key in status
    assert status["S"]["I"] == pytest.approx(round(cell.identity.I, 3))


def test_cell_death_modes_unregister_and_recycle_matter():
    nx1 = load_nx1()
    colony, rng = make_colony(nx1, seed=77)

    starving = add_cell(nx1, colony, rng, 2, 2, nx1.CellType.METABOLIC)
    starving.metabolism.a_free = 0.0
    starving._starving_ticks = 16
    starving._check_death()
    assert not starving.alive
    assert starving.death_cause == "M1_metabolica"
    assert not colony.world.is_occupied(2, 2)

    structural = add_cell(nx1, colony, rng, 4, 4, nx1.CellType.METABOLIC)
    structural.boundary.c_integrity = 0.01
    structural._check_death()
    assert not structural.alive
    assert structural.death_cause == "M2_structural"

    organizational = add_cell(nx1, colony, rng, 6, 6, nx1.CellType.METABOLIC)
    organizational.identity._history = [0.01] * 10
    organizational._check_death()
    assert not organizational.alive
    assert organizational.death_cause == "M3_organizacional"


def test_colony_resource_share_moves_resources_and_records_support():
    nx1 = load_nx1()
    colony, rng = make_colony(nx1, seed=88)
    donor = add_cell(nx1, colony, rng, 3, 3, nx1.CellType.METABOLIC)
    receiver = add_cell(nx1, colony, rng, 3, 4, nx1.CellType.REPAIR)
    donor.metabolism.a_free = 120.0
    receiver.metabolism.a_free = 5.0
    junction = colony._create_junction(nx1.JunctionKind.METABOLIC, donor.id, receiver.id, 0.9, 0.8, 0.2)
    receiver_before = receiver.metabolism.a_free

    colony._resource_share(donor, receiver, junction)

    assert receiver.metabolism.a_free > receiver_before
    assert donor.provided_support > 0.0
    assert receiver.received_support > 0.0
    assert junction.last_activity == colony.tick_count


def test_colony_status_exposes_cells_junctions_and_multiple_organisms():
    nx1 = load_nx1()
    colony, rng = make_colony(nx1, seed=99)
    a = add_cell(nx1, colony, rng, 1, 1, nx1.CellType.METABOLIC)
    b = add_cell(nx1, colony, rng, 1, 2, nx1.CellType.REPAIR)
    colony._create_junction(nx1.JunctionKind.ADHESION, a.id, b.id, 0.8, 0.1, 0.1)
    colony.organism.update(colony.cells, colony.junctions)

    status = colony.status()
    json.dumps(status)

    assert status["alive"] == 2
    assert status["junction_count"] == 1
    assert status["organism"]["organism_count"] == 1
    assert len(status["cells"]) == 2


def test_connected_components_become_distinct_organisms():
    nx1 = load_nx1()
    colony, rng = make_colony(nx1, seed=11)
    a = add_cell(nx1, colony, rng, 1, 1, nx1.CellType.METABOLIC)
    b = add_cell(nx1, colony, rng, 1, 2, nx1.CellType.REPAIR)
    c = add_cell(nx1, colony, rng, 12, 12, nx1.CellType.METABOLIC)
    d = add_cell(nx1, colony, rng, 12, 13, nx1.CellType.REPAIR)

    colony._create_junction(nx1.JunctionKind.ADHESION, a.id, b.id, 0.8, 0.1, 0.1)
    colony._create_junction(nx1.JunctionKind.ADHESION, c.id, d.id, 0.8, 0.1, 0.1)
    colony.organism.update(colony.cells, colony.junctions)

    organism_sizes = sorted(len(org.member_cell_ids) for org in colony.organism.organisms)
    organism_ids = {cell.organism_id for cell in (a, b, c, d)}

    assert organism_sizes == [2, 2]
    assert len(organism_ids) == 2
    assert colony.organism.to_dict()["organism_count"] == 2


def test_organism_metrics_use_body_boundary_exchange_and_neural_coordination():
    nx1 = load_nx1()
    colony, rng = make_colony(nx1, seed=22)
    boundary = add_cell(nx1, colony, rng, 5, 5, nx1.CellType.BOUNDARY)
    metabolic = add_cell(nx1, colony, rng, 5, 6, nx1.CellType.METABOLIC)
    repair = add_cell(nx1, colony, rng, 6, 6, nx1.CellType.REPAIR)
    signaling = add_cell(nx1, colony, rng, 6, 5, nx1.CellType.SIGNALING)
    neuron = add_cell(nx1, colony, rng, 7, 5, nx1.CellType.NEURON)
    sensory = add_cell(nx1, colony, rng, 7, 6, nx1.CellType.SENSORY)
    motor = add_cell(nx1, colony, rng, 8, 6, nx1.CellType.MOTOR)

    chain = [boundary, metabolic, repair, signaling, neuron, sensory, motor]
    for left, right in zip(chain, chain[1:]):
        colony._create_junction(nx1.JunctionKind.ADHESION, left.id, right.id, 0.9, 0.1, 0.1)
    colony._create_junction(nx1.JunctionKind.METABOLIC, metabolic.id, repair.id, 0.9, 0.8, 0.1)
    colony._create_junction(nx1.JunctionKind.SYNAPTIC, sensory.id, neuron.id, 0.8, 0.0, 0.8, 0.9)
    colony._create_junction(nx1.JunctionKind.SYNAPTIC, neuron.id, motor.id, 0.8, 0.0, 0.8, 0.9)

    colony.organism.update(colony.cells, colony.junctions)
    organism = colony.organism.organisms[0]

    assert organism.role_coverage == 1.0
    assert organism.boundary_closure > 0.0
    assert organism.metabolic_exchange > 0.0
    assert organism.neural_coordination > 0.0
    assert organism.collective_identity > 0.5


def test_synaptic_plasticity_tracks_organism_utility():
    nx1 = load_nx1()
    colony, rng = make_colony(nx1, seed=33)
    pre = add_cell(nx1, colony, rng, 5, 5, nx1.CellType.SENSORY)
    post = add_cell(nx1, colony, rng, 5, 6, nx1.CellType.MOTOR)
    colony._create_junction(nx1.JunctionKind.ADHESION, pre.id, post.id, 0.9, 0.1, 0.1)
    synapse = colony._create_junction(nx1.JunctionKind.SYNAPTIC, pre.id, post.id, 0.8, 0.0, 0.8, 0.4)
    colony.organism.update(colony.cells, colony.junctions)

    colony.tick_count = 12
    pre.spike_output = 0.9
    post.spike_output = 0.7
    initial_weight = synapse.weight

    colony._queue_synaptic_events()

    assert synapse.utility_trace > 0.0
    assert synapse.weight > initial_weight
    assert synapse.event_queue


def test_organism_reproduction_creates_seed_cluster_from_viable_body():
    nx1 = load_nx1()
    colony, rng = make_colony(nx1, max_cells=24, seed=44)
    roles = [
        nx1.CellType.BOUNDARY,
        nx1.CellType.METABOLIC,
        nx1.CellType.REPAIR,
        nx1.CellType.SIGNALING,
        nx1.CellType.NEURON,
        nx1.CellType.GERMLINE,
    ]
    cells = [
        add_cell(nx1, colony, rng, 8 + i % 3, 8 + i // 3, role)
        for i, role in enumerate(roles)
    ]
    for left, right in zip(cells, cells[1:]):
        colony._create_junction(nx1.JunctionKind.ADHESION, left.id, right.id, 0.95, 0.2, 0.2)
        colony._create_junction(nx1.JunctionKind.METABOLIC, left.id, right.id, 0.85, 0.7, 0.2)

    parent = cells[-1]
    parent.reproduction.r_maturity = 0.95
    parent.reproduction.build_offspring_genome = (
        lambda parent_genome, parent_neural, parent_memory, rng: parent_genome
    )
    colony.organism.update(colony.cells, colony.junctions)
    before_count = len(colony.cells)

    colony.tick_count = 40
    colony._try_organism_reproduction()

    created = [cell for cell in colony.cells.values() if cell._generation == parent._generation + 1]
    assert len(colony.cells) >= before_count + 3
    assert {cell.cell_type for cell in created[:3]} == {
        nx1.CellType.STEM,
        nx1.CellType.BOUNDARY,
        nx1.CellType.METABOLIC,
    }
    assert any("ORGANISM_REPRODUCTION" in event for event in parent.event_log)


def test_evolution_engine_status_is_json_serializable_after_tournament():
    nx1 = load_nx1()
    nx1.MAX_CELLS = 48
    rng = random.Random(55)
    np.random.seed(55)
    engine = nx1.EvolutionEngine(n_colonies=2, tournament_interval=4, rng=rng)

    for _ in range(8):
        engine.tick()

    status = engine.status()
    best_colony_status = engine.colonies[engine.best_colony_idx()].status()

    json.dumps(status)
    json.dumps(best_colony_status)
    assert status["tournaments_run"] >= 1
    assert status["evolution_history"]
    assert "best_organisms" in status["evolution_history"][-1]


def test_scale_error_paths_and_world_registration_snapshot():
    nx1 = load_nx1()
    with pytest.raises(ValueError):
        nx1.derive_simulation_scale(0)
    with pytest.raises(ValueError):
        nx1.derive_simulation_scale(nx1.MAX_DENSE_WORLD_CELLS + 1)

    colony, rng = make_colony(nx1, seed=120)
    a = add_cell(nx1, colony, rng, 2, 2, nx1.CellType.METABOLIC)
    b = add_cell(nx1, colony, rng, 3, 2, nx1.CellType.REPAIR)
    before = colony.world.sample_signals(2, 2).copy()
    colony.world.deposit_signal(2, 2, "energy_need", 0.0)
    neighbors = colony.world.get_neighbor_cells(2, 2, 2, colony.cells)
    snapshot = colony.world.snapshot()

    assert np.array_equal(before, colony.world.sample_signals(2, 2))
    assert neighbors[0]["id"] == b.id
    assert [a.x, a.y] in snapshot["occupied"]
    assert "signals" in snapshot


def test_low_level_branch_contracts_for_blocks_b1_to_b9():
    nx1 = load_nx1()
    genome = nx1.Genome(development_ticks=1, repr_min_age=1)

    boundary = nx1.Boundary(c_integrity=0.05)
    assert boundary.repair(0.0, 10.0, 1.0, genome) == (0.0, 0.0)
    assert boundary.is_dead

    met = nx1.Metabolism(a_free=2.0, m_struct=3.0, p_repair=4.0)
    assert met.consume_atp(5.0) == 2.0
    assert met.consume_structural(5.0) == 3.0
    assert met.consume_repair_material(5.0) == 4.0
    assert met.is_starving

    homeostasis = nx1.HomeostasisRegulator(genome)
    for i in range(25):
        homeostasis.update(nx1.Metabolism(a_free=60.0), nx1.Boundary(), i * 0.01, 0.8, np.ones(5) * 0.2)
    assert len(homeostasis._priority_history) == 20
    assert 0.0 <= homeostasis.g_regulatory_coherence <= 1.0

    neural = nx1.NeuralCore(genome)
    for _ in range(60):
        neural.step(np.ones(6) * 0.5, np.zeros(4), atp_available=20.0)
    w_ih, w_hh, w_ho = neural.get_heritable_predispositions()
    assert len(neural._input_history) == 50
    assert (len(w_ih), len(w_hh), len(w_ho)) == (24, 16, 12)

    memory = nx1.MaterialMemory(genome)
    corruption_before = memory.h_corruption_rate
    memory.update(0.7, {}, np.zeros(4), atp_available=0.0, waste_internal=5.0, damage=0.1)
    assert memory.h_corruption_rate > corruption_before

    identity = nx1.Identity()
    for _ in range(35):
        identity.compute(nx1.Boundary(), nx1.Metabolism(), nx1.MaterialMemory(genome),
                         nx1.HomeostasisRegulator(genome), nx1.NeuralCore(genome), damage=0.0)
    assert len(identity._history) == 30


def test_reproduction_inheritance_failure_and_development_failure_paths():
    nx1 = load_nx1()
    rng = random.Random(121)
    genome = nx1.Genome.create(rng)
    neural = nx1.NeuralCore(genome)
    memory = nx1.MaterialMemory(genome)
    reproduction = nx1.ReproductionModule(genome)

    memory.h_integrity = 0.2
    assert reproduction.build_offspring_genome(genome, neural, memory, rng) is None

    memory.h_integrity = 1.0
    child = reproduction.build_offspring_genome(genome, neural, memory, rng)
    assert child is not None
    assert child.W_reg().shape == (4, 5)
    assert child.neural_W_ih().shape == (4, 6)
    assert child.morphogen_response().shape == (2, 4)

    failing = nx1.ReproductionModule(nx1.Genome(development_ticks=1))
    failing._rng = random.Random(1)
    failing.r_failure_risk = 1.0
    failing.initiate()
    assert failing.develop_tick(nx1.Metabolism(a_free=50.0, m_struct=50.0), nx1.Genome(development_ticks=1)) == "fail"


def test_junction_organism_reset_and_synaptic_delivery_paths():
    nx1 = load_nx1()
    junction = nx1.Junction("j1", "a", "b", nx1.JunctionKind.ADHESION)
    organism = nx1.OrganismState()
    organism.update({}, {})
    assert junction.other("a") == "b"
    assert junction.other("b") == "a"
    assert junction.other("z") is None
    assert organism.development_stage == "extinct"
    assert organism.to_dict()["organism_count"] == 0

    colony, rng = make_colony(nx1, seed=122)
    pre = add_cell(nx1, colony, rng, 4, 4, nx1.CellType.NEURON)
    post = add_cell(nx1, colony, rng, 4, 5, nx1.CellType.MOTOR)
    synapse = colony._create_junction(nx1.JunctionKind.SYNAPTIC, pre.id, post.id, 0.8, 0.0, 0.8, 0.5)
    synapse.event_queue.append((0, 1.0))
    colony._deliver_synaptic_events()
    assert np.any(post.synaptic_input != 0.0)
    assert synapse.event_queue == []

    synapse.event_queue.append((10, 1.0))
    colony._deliver_synaptic_events()
    assert synapse.event_queue == [(10, 1.0)]

    post.alive = False
    colony._deliver_synaptic_events()
    assert synapse.id not in colony.junctions


def test_organism_pressure_and_reverse_resource_share_paths():
    nx1 = load_nx1()
    colony, rng = make_colony(nx1, seed=123)
    cells = [
        add_cell(nx1, colony, rng, 5, 5, nx1.CellType.BOUNDARY),
        add_cell(nx1, colony, rng, 5, 6, nx1.CellType.REPAIR),
        add_cell(nx1, colony, rng, 6, 6, nx1.CellType.NEURON),
    ]
    for left, right in zip(cells, cells[1:]):
        colony._create_junction(nx1.JunctionKind.ADHESION, left.id, right.id, 0.8, 0.1, 0.1)
    colony.organism.update(colony.cells, colony.junctions)
    org = colony.organism.organisms[0]
    org.collective_identity = 0.1
    org.shared_stress = 0.5
    before_damage = cells[0].damage_X
    colony._apply_organism_pressure()
    assert cells[0].damage_X > before_damage
    assert cells[0].homeostasis.p_repair >= 0.25

    a = add_cell(nx1, colony, rng, 8, 8, nx1.CellType.METABOLIC)
    b = add_cell(nx1, colony, rng, 8, 9, nx1.CellType.REPAIR)
    a.metabolism.a_free = 5.0
    b.metabolism.a_free = 120.0
    a.metabolism.w_waste = 1.0
    b.metabolism.w_waste = 60.0
    metabolic = colony._create_junction(nx1.JunctionKind.METABOLIC, a.id, b.id, 0.9, 0.8, 0.2)
    colony._resource_share(a, b, metabolic)
    assert a.metabolism.a_free > 5.0
    assert b.metabolism.w_waste < 60.0


def test_colony_limit_spawn_and_http_handler_paths():
    nx1 = load_nx1()
    colony, rng = make_colony(nx1, max_cells=1, seed=124)
    cell = add_cell(nx1, colony, rng, 1, 1, nx1.CellType.METABOLIC)
    extra = nx1.Cell(2, 2, colony.world, nx1.Genome.create(rng), rng=rng)
    assert not colony.add_cell(extra)
    assert cell.id in colony.cells

    empty, rng2 = make_colony(nx1, max_cells=4, seed=125)
    empty.spawn_primordial_from_genome(2, nx1.Genome.create(rng2))
    assert len(empty.cells) == 2

    previous_engine = nx1.EVOLUTION_ENGINE
    server = nx1.run_server(0)
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        nx1.EVOLUTION_ENGINE = None
        html = urllib.request.urlopen(base + "/", timeout=2).read().decode()
        loading_status = json.loads(urllib.request.urlopen(base + "/status", timeout=2).read().decode())
        loading_world = json.loads(urllib.request.urlopen(base + "/world", timeout=2).read().decode())
        loading_evo = json.loads(urllib.request.urlopen(base + "/evolution", timeout=2).read().decode())
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(base + "/missing", timeout=2)

        nx1.MAX_CELLS = 16
        nx1.EVOLUTION_ENGINE = nx1.EvolutionEngine(n_colonies=1, tournament_interval=10, rng=random.Random(126))
        engine_status = json.loads(urllib.request.urlopen(base + "/status", timeout=2).read().decode())
        engine_world = json.loads(urllib.request.urlopen(base + "/world", timeout=2).read().decode())
        engine_evo = json.loads(urllib.request.urlopen(base + "/evolution", timeout=2).read().decode())
    finally:
        nx1.EVOLUTION_ENGINE = previous_engine
        server.shutdown()
        server.server_close()

    assert "nx-1" in html
    assert loading_status["loading"]
    assert loading_world["loading"]
    assert loading_evo["loading"]
    assert err.value.code == 404
    assert "cells" in engine_status
    assert "nutrients" in engine_world
    assert engine_evo["n_colonies"] == 1


def test_cell_movement_refractory_and_reproduction_logging_branches(monkeypatch):
    nx1 = load_nx1()
    colony, rng = make_colony(nx1, seed=127)
    cell = add_cell(nx1, colony, rng, 10, 10, nx1.CellType.METABOLIC)

    cell.metabolism.a_free = 1.0
    old_pos = (cell.x, cell.y)
    cell._move({"move_x": 1.0, "move_y": 0.0}, nut_local=0.0, tox_local=0.0)
    assert (cell.x, cell.y) == old_pos

    cell.metabolism.a_free = 100.0
    cell.attachment_strength = 0.8
    cell.rng = random.Random(0)
    cell._move({"move_x": 1.0, "move_y": 0.0}, nut_local=0.0, tox_local=0.0)
    assert (cell.x, cell.y) == old_pos

    assert cell.tick() is None
    cell.alive = False
    assert cell.tick() is None

    for result, expected in [
        ("fail", "REPRODUCTION_FAILED:risk_exceeded"),
        ("abort", "REPRODUCTION_ABORTED:no_resources"),
    ]:
        cell.alive = True
        cell.reproduction._replicating = True
        monkeypatch.setattr(cell.reproduction, "develop_tick", lambda met, genome, result=result: result)
        cell.tick()
        assert any(expected in event for event in cell.event_log)


def test_cell_spawn_offspring_success_and_failure_paths(monkeypatch):
    nx1 = load_nx1()
    colony, rng = make_colony(nx1, seed=128)
    parent = add_cell(nx1, colony, rng, 5, 5, nx1.CellType.GERMLINE)
    monkeypatch.setattr(
        parent.reproduction,
        "build_offspring_genome",
        lambda parent_genome, parent_neural, parent_memory, rng: parent_genome,
    )

    child = parent._spawn_offspring()
    assert child is not None
    assert child._generation == parent._generation + 1
    assert colony.world.is_occupied(child.x, child.y)

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if (dx, dy) != (0, 0):
                colony.world.register(parent.x + dx, parent.y + dy, "blocked")
    assert parent._spawn_offspring() is None

    monkeypatch.setattr(
        parent.reproduction,
        "build_offspring_genome",
        lambda parent_genome, parent_neural, parent_memory, rng: None,
    )
    assert parent._spawn_offspring() is None


def test_junction_creation_maintenance_link_formation_and_pruning(monkeypatch):
    nx1 = load_nx1()
    colony, rng = make_colony(nx1, seed=129)
    a = add_cell(nx1, colony, rng, 2, 2, nx1.CellType.NEURON)
    b = add_cell(nx1, colony, rng, 2, 3, nx1.CellType.MOTOR)

    assert colony._create_junction(nx1.JunctionKind.ADHESION, a.id, a.id, 0.5, 0.1, 0.1) is None
    adhesion = colony._create_junction(nx1.JunctionKind.ADHESION, a.id, b.id, 0.9, 0.1, 0.1)
    assert colony._create_junction(nx1.JunctionKind.ADHESION, a.id, b.id, 0.9, 0.1, 0.1) is None
    assert adhesion is not None

    adhesion.age = 10
    a.metabolism.a_free = 0.0
    b.metabolism.a_free = 0.0
    old_strength = adhesion.strength
    colony._maintain_junctions_and_share()
    assert adhesion.strength < old_strength

    a.metabolism.a_free = 100.0
    b.metabolism.a_free = 100.0
    monkeypatch.setattr(colony.rng, "random", lambda: 0.0)
    colony.tick_count = 3
    colony._form_multicellular_links()
    assert any(j.kind == nx1.JunctionKind.SYNAPTIC for j in colony.junctions.values())

    for j in list(colony.junctions.values()):
        if j.kind == nx1.JunctionKind.SYNAPTIC:
            j.last_activity = -100
            j.prune_score = 1.1
    colony._maintain_junctions_and_share()
    assert all(j.kind != nx1.JunctionKind.SYNAPTIC for j in colony.junctions.values())


def test_main_startup_and_keyboard_interrupt_shutdown(monkeypatch, capsys):
    nx1 = load_nx1()

    class FakeServer:
        def __init__(self):
            self.shutdown_called = False

        def shutdown(self):
            self.shutdown_called = True

    class FakeCell:
        alive = True
        identity = type("IdentityProxy", (), {"I": 0.5})()
        _generation = 0

    class FakeColony:
        def __init__(self):
            self.cells = {"c": FakeCell()}
            self.organism = type("OrganismProxy", (), {"development_stage": "fake"})()

        def spawn_primordial_from_genome(self, n, genome):
            self.spawned = (n, genome)

    class FakeEvolutionEngine:
        def __init__(self, n_colonies, tournament_interval, rng):
            self.colonies = [FakeColony()]
            self.founding_genomes = [object()]
            self.evolution_history = []

        def tick(self):
            self._colony_fitness(self.colonies[0])
            self.colonies[0].spawn_primordial_from_genome(1, self.founding_genomes[0])
            raise KeyboardInterrupt

        def _colony_fitness(self, colony):
            return 1.0

    fake_server = FakeServer()
    monkeypatch.setattr(nx1, "run_server", lambda port: fake_server)
    monkeypatch.setattr(nx1, "EvolutionEngine", FakeEvolutionEngine)
    monkeypatch.setattr(sys, "argv", ["nx-1.py", "--max-cells", "8", "--n-colonies", "1", "--tournament-interval", "1", "--port", "0"])

    nx1.main()
    out = capsys.readouterr().out

    assert "nx-1" in out
    assert "Simulación detenida" in out
    assert fake_server.shutdown_called


def test_remaining_branch_paths_for_repair_reproduction_communication_and_main(monkeypatch, capsys):
    nx1 = load_nx1()
    genome = nx1.Genome(development_ticks=1, repr_min_age=1)

    boundary = nx1.Boundary(c_integrity=1.0)
    assert boundary.repair(10.0, 10.0, 1.0, genome) == (0.0, 0.0)
    damaged = nx1.Boundary(c_integrity=0.2)
    atp_used, mass_used = damaged.repair(0.001, 0.001, 1.0, genome)
    assert 0.0 < atp_used <= 0.001
    assert 0.0 < mass_used <= 0.001

    repair = nx1.RepairSystem(genome)
    repair.p_latency = 1
    low_budget = repair.execute(
        nx1.Metabolism(a_free=1.0, p_repair=1.0),
        nx1.Boundary(c_integrity=0.2),
        nx1.NeuralCore(genome),
        nx1.MaterialMemory(genome),
        nx1.HomeostasisRegulator(genome),
        damage=0.5,
        genome=genome,
    )
    assert low_budget == (0.0, {})

    reproduction = nx1.ReproductionModule(genome)
    assert reproduction.develop_tick(nx1.Metabolism(), genome) is None
    reproduction._replicating = True
    assert not reproduction.can_reproduce(
        nx1.Metabolism(a_free=100.0, m_struct=100.0),
        nx1.Boundary(),
        nx1.MaterialMemory(genome),
        nx1.HomeostasisRegulator(genome),
        0.0,
        1.0,
        genome,
    )

    rng = random.Random(130)
    world = nx1.SpatialWorld(width=12, height=12, n_sources=1, rng=rng)
    comm = nx1.CommunicationSystem(genome)
    for _ in range(20):
        world.deposit_signal(1, 1, "energy_need", 50.0)
        comm.receive(world, 1, 1, nx1.Boundary(), nx1.Metabolism(a_free=100.0),
                     nx1.HomeostasisRegulator(genome), nx1.MaterialMemory(genome), 0.0)
    assert len(comm.z_received_history) == 16

    org = nx1.OrganismState()
    assert org.get(None) is None
    assert org.get("missing") is None

    class FakeServer:
        def shutdown(self):
            self.shutdown_called = True

    class FakeCell:
        alive = False
        identity = type("IdentityProxy", (), {"I": 0.5})()
        _generation = 3

    class FakeColony:
        def __init__(self):
            self.cells = {"c": FakeCell()}
            self.organism = type("OrganismProxy", (), {"development_stage": "integrated"})()
            self.spawned = None

        def spawn_primordial_from_genome(self, n, genome):
            self.spawned = (n, genome)

    class SlowFakeEvolutionEngine:
        def __init__(self, n_colonies, tournament_interval, rng):
            self.colonies = [FakeColony()]
            self.founding_genomes = [object()]
            self.evolution_history = [{"best_fitness": 1.2, "mean_fitness": 0.8}]
            self.calls = 0

        def tick(self):
            self.calls += 1
            if self.calls > 100:
                raise KeyboardInterrupt

        def _colony_fitness(self, colony):
            return 1.0

    fake_server = FakeServer()
    monkeypatch.setattr(nx1, "run_server", lambda port: fake_server)
    monkeypatch.setattr(nx1, "EvolutionEngine", SlowFakeEvolutionEngine)
    monkeypatch.setattr(nx1.time, "sleep", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["nx-1.py", "--max-cells", "8", "--n-colonies", "1"])

    nx1.main()
    out = capsys.readouterr().out
    assert "t=   100" in out
    assert "Último torneo" in out


def test_neighbor_query_filters_empty_missing_and_dead_cells():
    nx1 = load_nx1()
    rng = random.Random(131)
    world = nx1.SpatialWorld(width=12, height=12, n_sources=0, rng=rng)
    live = nx1.Cell(2, 2, world, nx1.Genome.create(rng), rng=random.Random(1))
    dead = nx1.Cell(2, 3, world, nx1.Genome.create(rng), rng=random.Random(2))
    dead.alive = False
    world.occupied[(3, 2)] = "missing-cell"
    world.occupied[(1, 2)] = live.id
    world.occupied[(2, 1)] = live.id

    neighbors = world.get_neighbor_cells(2, 2, 1, {live.id: live, dead.id: dead})

    assert len(neighbors) == 1
    assert neighbors[0]["id"] == live.id


def test_organism_stage_classification_reaches_aggregate_and_integrated_body():
    nx1 = load_nx1()
    colony, rng = make_colony(nx1, max_cells=16, seed=132)
    low_a = add_cell(nx1, colony, rng, 1, 1, nx1.CellType.STEM, identity=0.05)
    low_b = add_cell(nx1, colony, rng, 1, 2, nx1.CellType.STEM, identity=0.05)
    low_c = add_cell(nx1, colony, rng, 2, 1, nx1.CellType.STEM, identity=0.05)
    colony._create_junction(nx1.JunctionKind.ADHESION, low_a.id, low_b.id, 0.08, 0.0, 0.0)
    colony._create_junction(nx1.JunctionKind.ADHESION, low_b.id, low_c.id, 0.08, 0.0, 0.0)

    aggregate = colony.organism._compute_instance(
        "aggregate",
        {low_a.id, low_b.id, low_c.id},
        colony.cells,
        colony.junctions,
    )

    assert aggregate.development_stage == "aggregate"

    colony, rng = make_colony(nx1, max_cells=16, seed=133)
    roles = [
        nx1.CellType.BOUNDARY,
        nx1.CellType.BOUNDARY,
        nx1.CellType.METABOLIC,
        nx1.CellType.REPAIR,
        nx1.CellType.SIGNALING,
        nx1.CellType.NEURON,
        nx1.CellType.SENSORY,
        nx1.CellType.MOTOR,
    ]
    cells = [
        add_cell(nx1, colony, rng, 4 + (idx % 4), 4 + (idx // 4), role, identity=0.95)
        for idx, role in enumerate(roles)
    ]
    for c in cells:
        if c.cell_type == nx1.CellType.BOUNDARY:
            c.boundary.c_integrity = 1.0
    for first, second in zip(cells, cells[1:]):
        colony._create_junction(nx1.JunctionKind.ADHESION, first.id, second.id, 1.0, 0.0, 0.1)
    colony._create_junction(nx1.JunctionKind.METABOLIC, cells[2].id, cells[3].id, 1.0, 1.0, 0.2)
    colony._create_junction(nx1.JunctionKind.SYNAPTIC, cells[5].id, cells[7].id, 1.0, 0.0, 1.0, 1.8)
    colony._create_junction(nx1.JunctionKind.SYNAPTIC, cells[6].id, cells[5].id, 1.0, 0.0, 1.0, 1.8)
    colony._create_junction(nx1.JunctionKind.SYNAPTIC, cells[5].id, cells[3].id, 1.0, 0.0, 1.0, 1.8)
    colony._create_junction(nx1.JunctionKind.SYNAPTIC, cells[6].id, cells[7].id, 1.0, 0.0, 1.0, 1.8)

    integrated = colony.organism._compute_instance(
        "integrated",
        {c.id for c in cells},
        colony.cells,
        colony.junctions,
    )

    assert integrated.development_stage == "integrated_body"


def test_cell_tick_covers_causal_damage_synaptic_refractory_spike_and_reproduction(monkeypatch):
    nx1 = load_nx1()
    colony, rng = make_colony(nx1, seed=134)
    cell = add_cell(nx1, colony, rng, 5, 5, nx1.CellType.GERMLINE, identity=0.95)
    cell.identity.i_causal_closure_proxy = 0.0
    cell.synaptic_input = np.array([0.2, -0.1, 0.05, 0.03])
    cell._refractory_ticks = 1
    child = nx1.Cell(6, 5, colony.world, nx1.Genome.create(rng), rng=random.Random(3), developing=True)
    monkeypatch.setattr(cell, "_move", lambda neural_out, nut_local, tox_local: None)
    monkeypatch.setattr(cell.reproduction, "can_reproduce", lambda *args: True)
    monkeypatch.setattr(cell.reproduction, "develop_tick", lambda *args: "success")
    monkeypatch.setattr(cell, "_spawn_offspring", lambda: child)

    offspring = cell.tick()

    assert offspring is child
    assert cell.spike_output == 0.0
    assert cell.synaptic_input[0] == pytest.approx(0.07)
    assert any("REPRODUCTION_INITIATED" in event for event in cell.event_log)
    assert any(f"OFFSPRING:{child.id}" in event for event in cell.event_log)

    spiker = add_cell(nx1, colony, rng, 7, 7, nx1.CellType.NEURON, identity=0.95)

    def force_spike(inputs, adaptive_bias, atp_available):
        spiker.neural.t_excitation = nx1.EXCIT_THRESHOLD + 0.4
        return {"move_x": 0.0, "move_y": 0.0, "capture_modulation": 0.5}

    monkeypatch.setattr(spiker.neural, "step", force_spike)
    monkeypatch.setattr(spiker, "_move", lambda neural_out, nut_local, tox_local: None)
    monkeypatch.setattr(spiker.reproduction, "can_reproduce", lambda *args: False)

    spiker.tick()

    assert spiker.spike_output > 0.0
    assert spiker._refractory_ticks == nx1.REFRACTORY_PERIOD

    failed_parent = add_cell(nx1, colony, rng, 8, 8, nx1.CellType.GERMLINE, identity=0.95)
    monkeypatch.setattr(failed_parent, "_move", lambda neural_out, nut_local, tox_local: None)
    monkeypatch.setattr(failed_parent.reproduction, "can_reproduce", lambda *args: True)
    monkeypatch.setattr(failed_parent.reproduction, "develop_tick", lambda *args: "success")
    monkeypatch.setattr(failed_parent, "_spawn_offspring", lambda: None)

    assert failed_parent.tick() is None
    assert any("OFFSPRING_FAILED:genome_corrupted" in event for event in failed_parent.event_log)


def test_movement_death_phase_log_and_metric_branches(monkeypatch):
    nx1 = load_nx1()
    colony, rng = make_colony(nx1, seed=135)
    cell = add_cell(nx1, colony, rng, 4, 4, nx1.CellType.MOTOR)
    cell.genome.motility = 0.0
    cell.genome.chemotaxis_gain = 0.0
    cell.genome.toxin_avoidance = 0.0
    monkeypatch.setattr(cell.rng, "random", lambda: 0.0)
    monkeypatch.setattr(cell.rng, "choice", lambda values: values[-1])
    cell._move({"move_x": 0.0, "move_y": 0.0}, 0.0, 0.0)
    assert (cell.x, cell.y) == (5, 5)

    blocked = add_cell(nx1, colony, rng, 6, 6, nx1.CellType.MOTOR)
    blocked.genome.motility = 0.0
    blocked.genome.chemotaxis_gain = 0.0
    blocked.genome.toxin_avoidance = 0.0
    monkeypatch.setattr(blocked.rng, "random", lambda: 0.0)
    monkeypatch.setattr(blocked.rng, "choice", lambda values: values[-1])
    monkeypatch.setattr(blocked.metabolism, "consume_atp", lambda amount: 0.0)
    blocked._move({"move_x": 0.0, "move_y": 0.0}, 0.0, 0.0)
    assert (blocked.x, blocked.y) == (6, 6)

    intoxicated = add_cell(nx1, colony, rng, 9, 9, nx1.CellType.METABOLIC)
    intoxicated.metabolism.w_waste = intoxicated.metabolism.w_waste_cap
    intoxicated._check_death()
    assert intoxicated.death_cause == "M1_intoxicacion"

    damaged = add_cell(nx1, colony, rng, 10, 10, nx1.CellType.REPAIR)
    damaged.damage_X = 0.99
    damaged._check_death()
    assert damaged.death_cause == "M2_damage"

    phases = [
        ({"alive": False}, nx1.LifePhase.ACTIVE),
        ({"age_ticks": 0}, nx1.LifePhase.DEVELOPING),
        ({"age_ticks": 999, "replicating": True}, nx1.LifePhase.REPLICATING),
        ({"age_ticks": 999, "stress": 0.8}, nx1.LifePhase.STRESSED),
        ({"age_ticks": 999, "damage": 0.25}, nx1.LifePhase.REPAIRING),
        ({"age_ticks": 500, "coherence": 0.2}, nx1.LifePhase.AGING),
        ({"age_ticks": 999, "identity": 0.1}, nx1.LifePhase.DYING),
    ]
    for idx, (setup, expected) in enumerate(phases):
        c = add_cell(nx1, colony, rng, idx, 11, nx1.CellType.STEM)
        c.age_ticks = setup.get("age_ticks", 999)
        c.alive = setup.get("alive", True)
        c.reproduction._replicating = setup.get("replicating", False)
        c.homeostasis.g_stress = setup.get("stress", 0.0)
        c.damage_X = setup.get("damage", 0.0)
        c.identity.i_structural_coherence = setup.get("coherence", 0.9)
        c.identity.I = setup.get("identity", 0.9)
        c._update_phase()
        assert c.phase == expected

    for i in range(51):
        cell._log(f"event-{i}")
    assert len(cell.event_log) == 50
    cell.age_ticks = 10
    cell.metrics_history = [{"t": i} for i in range(60)]
    cell._record_metrics()
    assert len(cell.metrics_history) == 60
    assert cell.metrics_history[-1]["t"] == 10


def test_colony_spawn_tick_and_junction_branch_paths(monkeypatch):
    nx1 = load_nx1()
    colony, rng = make_colony(nx1, max_cells=8, seed=136)
    cell = nx1.Cell(1, 1, colony.world, nx1.Genome.create(rng), rng=random.Random(1))
    assert colony.add_cell(cell)

    colony.spawn_primordial(1)
    assert len(colony.cells) >= 2

    no_source_colony, no_source_rng = make_colony(nx1, max_cells=8, seed=137)
    no_source_colony.world.sources = []
    no_source_colony.spawn_primordial(1)
    no_source_colony.spawn_primordial_from_genome(1, nx1.Genome.create(no_source_rng))
    assert len(no_source_colony.cells) == 2

    tick_colony, tick_rng = make_colony(nx1, max_cells=4, seed=138)
    parent = add_cell(nx1, tick_colony, tick_rng, 2, 2, nx1.CellType.METABOLIC)
    predead = add_cell(nx1, tick_colony, tick_rng, 2, 3, nx1.CellType.METABOLIC)
    predead.alive = False
    predead.death_cause = "predead"
    offspring = nx1.Cell(3, 2, tick_colony.world, nx1.Genome.create(tick_rng), rng=random.Random(2))
    tick_colony.dead_log = [{"id": str(i)} for i in range(100)]

    def parent_tick():
        parent.alive = False
        parent.death_cause = "test"
        return offspring

    monkeypatch.setattr(parent, "tick", parent_tick)
    for name in (
        "_deliver_synaptic_events",
        "_maintain_junctions_and_share",
        "_update_attachment_strengths",
        "_form_multicellular_links",
        "_differentiate_cells",
        "_queue_synaptic_events",
        "_police_cells",
        "_apply_organism_pressure",
        "_try_organism_reproduction",
    ):
        monkeypatch.setattr(tick_colony, name, lambda: None)
    monkeypatch.setattr(tick_colony.organism, "update", lambda cells, junctions: None)

    status = tick_colony.tick(include_status=True)

    assert parent.id not in tick_colony.cells
    assert offspring.id in tick_colony.cells
    assert len(tick_colony.dead_log) == 100
    assert status["alive"] == 1

    junction_colony, junction_rng = make_colony(nx1, max_cells=8, seed=139)
    a = add_cell(nx1, junction_colony, junction_rng, 4, 4, nx1.CellType.NEURON)
    b = add_cell(nx1, junction_colony, junction_rng, 4, 5, nx1.CellType.MOTOR)
    assert junction_colony._remove_junction("missing") is None
    adhesion = junction_colony._create_junction(nx1.JunctionKind.ADHESION, a.id, b.id, 0.8, 0.1, 0.1)
    junction_colony._remove_cell_junctions(a.id)
    assert adhesion.id not in junction_colony.junctions

    one_cell_colony, one_rng = make_colony(nx1, seed=140)
    add_cell(nx1, one_cell_colony, one_rng, 1, 1, nx1.CellType.STEM)
    one_cell_colony.tick_count = 3
    assert one_cell_colony._form_multicellular_links() is None

    capped_colony, capped_rng = make_colony(nx1, seed=141)
    capped_a = add_cell(nx1, capped_colony, capped_rng, 1, 1, nx1.CellType.STEM)
    add_cell(nx1, capped_colony, capped_rng, 1, 2, nx1.CellType.STEM)
    capped_a.junction_ids = {str(i) for i in range(8)}
    capped_colony.tick_count = 3
    assert capped_colony._form_multicellular_links() is None

    other_capped, oc_rng = make_colony(nx1, seed=142)
    oc_a = add_cell(nx1, other_capped, oc_rng, 2, 2, nx1.CellType.STEM)
    oc_b = add_cell(nx1, other_capped, oc_rng, 2, 3, nx1.CellType.STEM)
    oc_b.junction_ids = {str(i) for i in range(8)}
    other_capped.tick_count = 3
    assert other_capped._form_multicellular_links() is None

    syn_only_colony, syn_rng = make_colony(nx1, seed=156)
    syn_a = add_cell(nx1, syn_only_colony, syn_rng, 4, 4, nx1.CellType.NEURON)
    syn_b = add_cell(nx1, syn_only_colony, syn_rng, 4, 5, nx1.CellType.MOTOR)
    syn_only_colony._create_junction(nx1.JunctionKind.SYNAPTIC, syn_a.id, syn_b.id, 0.7, 0.0, 0.5, 0.2)
    monkeypatch.setattr(syn_only_colony.rng, "random", lambda: 1.0)
    syn_only_colony.tick_count = 3
    assert syn_only_colony._form_multicellular_links() is None

    link_colony, link_rng = make_colony(nx1, seed=143)
    link_a = add_cell(nx1, link_colony, link_rng, 3, 3, nx1.CellType.NEURON)
    link_b = add_cell(nx1, link_colony, link_rng, 3, 4, nx1.CellType.MOTOR)
    weak = link_colony._create_junction(nx1.JunctionKind.ADHESION, link_a.id, link_b.id, 0.2, 0.1, 0.1)
    monkeypatch.setattr(link_colony.rng, "random", lambda: 1.0)
    link_colony.tick_count = 3
    assert link_colony._form_multicellular_links() is None
    weak.age = 10
    weak.strength = 0.8
    del link_colony.cells[link_b.id]
    add_cell(nx1, link_colony, random.Random(201), 8, 8, nx1.CellType.STEM)
    add_cell(nx1, link_colony, random.Random(202), 9, 8, nx1.CellType.STEM)
    assert link_colony._form_multicellular_links() is None

    metabolic_colony, metabolic_rng = make_colony(nx1, seed=144)
    ma = add_cell(nx1, metabolic_colony, metabolic_rng, 5, 5, nx1.CellType.NEURON)
    mb = add_cell(nx1, metabolic_colony, metabolic_rng, 5, 6, nx1.CellType.MOTOR)
    ma.metabolism.a_free = ma.metabolism.a_free_cap
    mb.metabolism.a_free = 1.0
    ma.damage_X = 0.0
    mb.damage_X = 0.8
    strong = metabolic_colony._create_junction(nx1.JunctionKind.ADHESION, ma.id, mb.id, 0.9, 0.1, 0.1)
    strong.age = 10
    monkeypatch.setattr(metabolic_colony.rng, "random", lambda: 0.0)
    metabolic_colony.tick_count = 3
    metabolic_colony._form_multicellular_links()
    assert any(j.kind == nx1.JunctionKind.METABOLIC for j in metabolic_colony.junctions.values())


def test_resource_sharing_synapses_differentiation_policing_and_pressure(monkeypatch):
    nx1 = load_nx1()
    colony, rng = make_colony(nx1, max_cells=20, seed=145)
    a = add_cell(nx1, colony, rng, 2, 2, nx1.CellType.METABOLIC)
    b = add_cell(nx1, colony, rng, 2, 3, nx1.CellType.REPAIR)
    zero = colony._create_junction(nx1.JunctionKind.METABOLIC, a.id, b.id, 0.5, 0.0, 0.1)
    colony._resource_share(a, b, zero)
    colony._remove_junction(zero.id)

    metabolic = colony._create_junction(nx1.JunctionKind.METABOLIC, b.id, a.id, 0.9, 0.8, 0.1)
    a.metabolism.a_free = a.metabolism.a_free_cap
    b.metabolism.a_free = 1.0
    a.metabolism.r_raw = 100.0
    b.metabolism.r_raw = 1.0
    a.metabolism.p_repair = 50.0
    b.metabolism.p_repair = 1.0
    a.damage_X = 0.0
    b.damage_X = 0.7
    a.metabolism.w_waste = a.metabolism.w_waste_cap
    b.metabolism.w_waste = 0.0
    colony._resource_share(a, b, metabolic)
    assert b.received_support > 0.0

    a.metabolism.a_free = 1.0
    b.metabolism.a_free = b.metabolism.a_free_cap
    b.metabolism.r_raw = 100.0
    a.metabolism.r_raw = 1.0
    b.metabolism.p_repair = 50.0
    a.metabolism.p_repair = 1.0
    b.damage_X = 0.0
    a.damage_X = 0.7
    b.metabolism.w_waste = b.metabolism.w_waste_cap
    a.metabolism.w_waste = 0.0
    colony._resource_share(a, b, metabolic)
    assert a.received_support > 0.0

    a.metabolism.r_raw = 200.0
    b.metabolism.r_raw = b.metabolism.r_raw_cap
    colony._resource_share(a, b, metabolic)
    assert b.metabolism.r_raw == b.metabolism.r_raw_cap

    colony._maintain_junctions_and_share()
    assert metabolic.last_activity == colony.tick_count

    b.alive = False
    doomed = colony._create_junction(nx1.JunctionKind.ADHESION, a.id, b.id, 0.5, 0.1, 0.1)
    colony._maintain_junctions_and_share()
    assert doomed.id not in colony.junctions
    b.alive = True

    syn = colony._create_junction(nx1.JunctionKind.SYNAPTIC, a.id, b.id, 0.8, 0.0, 0.9, 0.5)
    b.alive = False
    colony._queue_synaptic_events()
    assert syn.id not in colony.junctions

    b.alive = True
    syn = colony._create_junction(nx1.JunctionKind.SYNAPTIC, a.id, b.id, 0.8, 0.0, 0.9, 0.5)
    a.organism_id = b.organism_id = "org"
    a.cell_type = nx1.CellType.METABOLIC
    b.cell_type = nx1.CellType.BOUNDARY
    syn.last_pre_tick = 10
    syn.last_post_tick = 8
    syn.utility_trace = -0.3
    colony.organism.organisms = [
        nx1.OrganismInstance(
            organism_id="org",
            member_cell_ids={a.id, b.id},
            junction_ids={syn.id},
            shared_stress=0.0,
            boundary_integrity=0.5,
            boundary_closure=0.5,
            collective_energy_pressure=1.0,
            collective_damage=1.0,
            collective_identity=0.0,
            development_stage="aggregate",
            reproduction_pressure=0.0,
            role_coverage=0.0,
            topology_integrity=0.0,
            metabolic_exchange=0.0,
            neural_coordination=0.0,
        )
    ]
    before_prune = syn.prune_score
    colony._queue_synaptic_events()
    assert syn.prune_score > before_prune

    diff = add_cell(nx1, colony, rng, 6, 6, nx1.CellType.STEM)
    diff.organism_id = None
    diff.age_ticks = 999
    diff.type_commitment = 0.0
    diff.metabolism.a_free = diff.metabolism.a_free_cap
    diff.metabolism.r_raw = diff.metabolism.r_raw_cap
    colony.tick_count = 5
    colony._differentiate_cells()
    assert diff.cell_type != nx1.CellType.STEM

    cheater = add_cell(nx1, colony, rng, 7, 7, nx1.CellType.POLICING)
    cheater.received_support = 5.0
    cheater.provided_support = 0.0
    cheater.junction_ids = {"j1", "j2"}
    colony.junctions["j1"] = nx1.Junction("j1", cheater.id, a.id, nx1.JunctionKind.ADHESION)
    colony.junctions["j2"] = nx1.Junction("j2", cheater.id, b.id, nx1.JunctionKind.ADHESION)
    colony._police_cells()
    cheater.cheater_score = 0.8
    colony._police_cells()
    assert cheater.cheater_score < 0.8
    assert colony.junctions["j1"].strength < 1.0

    missing_id = "missing"
    pressure_cells = [
        add_cell(nx1, colony, rng, 8, 8, nx1.CellType.BOUNDARY),
        add_cell(nx1, colony, rng, 8, 9, nx1.CellType.NEURON),
        add_cell(nx1, colony, rng, 9, 8, nx1.CellType.REPAIR),
    ]
    dead_member = pressure_cells[2]
    dead_member.alive = False
    org_zero = nx1.OrganismInstance("zero", {pressure_cells[0].id}, set(), 0, 0, 0, 0, 0, 0, "aggregate", 0, 0, 0, 0, 0)
    org_low = nx1.OrganismInstance(
        "low",
        {c.id for c in pressure_cells} | {missing_id},
        set(),
        shared_stress=0.5,
        boundary_integrity=0.9,
        boundary_closure=0.1,
        collective_energy_pressure=0.1,
        collective_damage=0.1,
        collective_identity=0.1,
        development_stage="aggregate",
        reproduction_pressure=0.0,
        role_coverage=0.5,
        topology_integrity=0.5,
        metabolic_exchange=0.0,
        neural_coordination=0.2,
    )
    org_high = nx1.OrganismInstance(
        "high",
        {pressure_cells[0].id, pressure_cells[1].id},
        set(),
        shared_stress=0.5,
        boundary_integrity=0.9,
        boundary_closure=0.1,
        collective_energy_pressure=0.1,
        collective_damage=0.1,
        collective_identity=0.6,
        development_stage="integrated_body",
        reproduction_pressure=0.0,
        role_coverage=1.0,
        topology_integrity=0.5,
        metabolic_exchange=0.0,
        neural_coordination=0.2,
    )
    colony.organism.organisms = [org_zero, org_low, org_high]
    before_boundary_score = pressure_cells[0].contribution_score
    before_neuron_score = pressure_cells[1].contribution_score
    colony._apply_organism_pressure()
    assert pressure_cells[0].homeostasis.p_repair > 0.0
    assert pressure_cells[0].contribution_score > before_boundary_score
    assert pressure_cells[1].contribution_score > before_neuron_score


def test_organism_reproduction_and_evolution_edge_paths(monkeypatch):
    nx1 = load_nx1()

    def reproductive_colony(seed=146, max_cells=16):
        colony, rng = make_colony(nx1, max_cells=max_cells, seed=seed)
        parent = add_cell(nx1, colony, rng, 6, 6, nx1.CellType.GERMLINE, identity=0.95)
        parent.reproduction.r_maturity = 0.9
        parent.metabolism.a_free = 80.0
        parent.metabolism.m_struct = 80.0
        members = {parent.id}
        for idx, role in enumerate([nx1.CellType.BOUNDARY, nx1.CellType.METABOLIC, nx1.CellType.REPAIR]):
            members.add(add_cell(nx1, colony, rng, 7 + idx, 6, role, identity=0.9).id)
        org = nx1.OrganismInstance(
            "parent-org",
            members,
            set(),
            shared_stress=0.0,
            boundary_integrity=0.9,
            boundary_closure=0.4,
            collective_energy_pressure=0.1,
            collective_damage=0.0,
            collective_identity=0.7,
            development_stage="integrated",
            reproduction_pressure=0.2,
            role_coverage=0.9,
            topology_integrity=0.8,
            metabolic_exchange=0.4,
            neural_coordination=0.0,
        )
        colony.organism.organisms = [org]
        colony.tick_count = 40
        return colony, rng, parent, org

    full, _, _, _ = reproductive_colony(seed=147, max_cells=4)
    assert full._try_organism_reproduction() is None

    no_candidates, _, _, _ = reproductive_colony(seed=148)
    no_candidates.organism.organisms = []
    assert no_candidates._try_organism_reproduction() is None

    neural_capacity, _, _, neural_org = reproductive_colony(seed=149, max_cells=7)
    neural_org.neural_coordination = 0.3
    assert neural_capacity._try_organism_reproduction() is None

    no_germline, _, parent, _ = reproductive_colony(seed=150)
    parent.cell_type = nx1.CellType.STEM
    assert no_germline._try_organism_reproduction() is None

    no_space, _, _, _ = reproductive_colony(seed=151)
    monkeypatch.setattr(no_space, "_free_positions_near", lambda x, y, radius, limit: [])
    assert no_space._try_organism_reproduction() is None

    heredity_fail, _, parent, _ = reproductive_colony(seed=152)
    monkeypatch.setattr(parent.reproduction, "build_offspring_genome", lambda *args: None)
    heredity_fail._try_organism_reproduction()
    assert any("ORGANISM_REPRO_ABORTED:heredity" in event for event in parent.event_log)

    atp_fail, _, parent, _ = reproductive_colony(seed=153)
    monkeypatch.setattr(parent.metabolism, "consume_atp", lambda amount: 0.0)
    assert atp_fail._try_organism_reproduction() is None

    struct_fail, _, parent, _ = reproductive_colony(seed=154)
    monkeypatch.setattr(parent.metabolism, "consume_atp", lambda amount: amount)
    monkeypatch.setattr(parent.metabolism, "consume_structural", lambda amount: 0.0)
    assert struct_fail._try_organism_reproduction() is None

    success, _, parent, org = reproductive_colony(seed=155, max_cells=20)
    org.neural_coordination = 0.3
    monkeypatch.setattr(success.rng, "random", lambda: 0.0)
    before = len(success.cells)
    success._try_organism_reproduction()
    assert len(success.cells) == before + 5
    assert any("ORGANISM_REPRODUCTION:parent-org:seed_cluster" in event for event in parent.event_log)
    assert any(j.kind == nx1.JunctionKind.SYNAPTIC and j.receptor_type == "inhibitory" for j in success.junctions.values())

    empty_engine = nx1.EvolutionEngine(n_colonies=2, tournament_interval=10, rng=random.Random(156))
    empty_engine.colonies[0].cells.clear()
    assert empty_engine._colony_fitness(empty_engine.colonies[0]) == 0.0
    empty_engine.colonies[1].cells.clear()
    empty_engine.evolution_history = [{"tournament": i} for i in range(20)]
    resets = []
    monkeypatch.setattr(empty_engine, "_reset_colony", lambda idx, genome: resets.append((idx, genome)))
    empty_engine._run_tournament()
    assert resets
    assert len(empty_engine.evolution_history) == 20
