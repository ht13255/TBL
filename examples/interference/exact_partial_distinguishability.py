"""Three photons with a measured complex internal-state Gram matrix."""

import numpy as np

import tbl

rng = np.random.default_rng(5)
raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
unitary, _ = np.linalg.qr(raw)

internal_states = np.array(
    [[1, 0, 0], [0.7, np.sqrt(0.51), 0], [0.3, 0.4j, np.sqrt(0.75)]],
    dtype=complex,
)
overlap_matrix = internal_states.conj() @ internal_states.T

circuit = tbl.LinearOpticalCircuit.from_transfer_matrix(unitary)
distribution = tbl.FockSimulator(circuit).probabilities([1, 1, 1], overlap_matrix=overlap_matrix)
for occupation, probability in sorted(distribution.probabilities.items()):
    print(occupation, f"{probability:.8f}")
