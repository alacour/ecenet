"""ASE Calculator for ECENet.

Usage example:

    from ase.io import read
    from ase import units
    from ase.md.langevin import Langevin
    from ecenet.calculator import ECENetCalculator

    calc = ECENetCalculator.from_checkpoint('molecule.mdl')

    atoms = read('start.xyz')
    atoms.calc = calc

    # Single-point
    print(atoms.get_potential_energy())   # eV
    print(atoms.get_forces())             # eV/Å

    # MD
    dyn = Langevin(atoms, timestep=0.5*units.fs,
                   temperature_K=300, friction=0.01/units.fs)
    dyn.run(1000)
"""

import time

import numpy as np
import torch
from ase import units as ase_units
from ase.calculators.calculator import Calculator, all_changes

# Conversion: 1 kcal/mol → eV
_KCAL_MOL_TO_EV = ase_units.kcal / ase_units.mol


class ECENetCalculator(Calculator):
    """ASE calculator wrapping ECENet.

    Parameters
    ----------
    model : ECENet
        Trained model (already on the correct device/dtype).
    device : torch.device
    dtype  : torch.dtype
    energy_reference : dict or None
        Optional per-element reference energies {symbol: eV} to add back
        to the model's residual energy (needed if model was trained on
        residual energies).  Keys are element symbols, values are eV/atom.
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, model, device=None, dtype=torch.float64,
                 energy_reference=None, element_to_type=None,
                 energy_units='eV', energy_mean=0.0,
                 log_timings=False, **kwargs):
        super().__init__(**kwargs)
        self._mic_warned = False
        self.model = model
        self.model.eval()
        # Ensure the analytic ACE basis is used (no SH in the backward graph).
        self.model.analytic_ace_basis = True
        self.dtype  = dtype
        self.device = device or next(model.parameters()).device
        self.log_timings = log_timings
        self._step_count = 0
        # energy_reference: {symbol: eV/atom} — added to predicted energy
        self.energy_reference = energy_reference or {}
        # element_to_type: {symbol: int} mapping — required for calculate().
        self.element_to_type = element_to_type or {}
        # unit conversion: model output → eV (and eV/Å for forces)
        if energy_units == 'kcal/mol':
            self._to_ev = _KCAL_MOL_TO_EV
        else:
            self._to_ev = 1.0
        # training mean energy (already in model units) converted to eV
        self._energy_mean_ev = energy_mean * self._to_ev

    # ── Construction helpers ────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None, dtype=None,
                        energy_reference=None, element_to_type=None,
                        energy_units=None, log_timings=False):
        """Load model and hparams directly from a checkpoint file.

        The checkpoint is expected to be self-describing: the training scripts
        store the architecture (``hparams``), an element mapping, and any unit /
        reference-energy metadata. No dataset-specific knowledge lives here.

        Parameters
        ----------
        checkpoint_path : str
            Path to a .mdl checkpoint saved by an ECENet training script.
        device : str or torch.device, optional
            Defaults to CUDA if available, else CPU.
        dtype : torch.dtype, optional
            Defaults to float32 if checkpoint was trained with float32,
            float64 otherwise (inferred from stored weights).
        energy_reference : dict, optional
            Per-element reference energies {symbol: eV}. If None, taken from the
            checkpoint's 'energy_reference' dict, or built from an 'e_ref' array
            indexed by the checkpoint's own element mapping.
        element_to_type : dict, optional
            {symbol: int} mapping override. If None, read from the checkpoint
            ('element_to_type', or 'type_to_idx' keyed by atomic number).
        energy_units : str, optional
            'eV' or 'kcal/mol'. If None, read from the checkpoint's
            'energy_units' key, defaulting to 'eV'.
        """
        from ecenet import ECENet

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        hp = ckpt.get('hparams')
        if hp is None:
            raise ValueError(
                "Checkpoint does not contain 'hparams'; re-save it with a "
                "current training script (which stores the architecture)."
            )

        # Infer dtype from stored weights
        if dtype is None:
            state = ckpt.get('best_state') or ckpt.get('model')
            sample = next(iter(state.values()))
            dtype = sample.dtype

        # n_mp (number of equivariant-layer stages) is passed to the constructor
        # separately; the rest of hparams maps directly onto ECENet's signature.
        hp = dict(hp)  # copy so we can pop
        n_mp = hp.pop('n_mp', 1)

        model = ECENet(**hp, n_mp=n_mp)
        if dtype == torch.float64:
            model = model.double()
        model = model.to(device)

        state = ckpt.get('best_state') or ckpt.get('model')
        model.load_state_dict(state, strict=False)
        model.eval()

        # ── Self-describing metadata (no dataset-specific knowledge here) ─────
        # Training scripts write these generic keys; the calculator just reads
        # them. Explicit arguments always win over the checkpoint.

        # Element → type-index mapping. Prefer an explicit symbol-keyed map;
        # otherwise convert a 'type_to_idx' that is keyed by atomic number.
        if element_to_type is None:
            element_to_type = ckpt.get('element_to_type')
        if element_to_type is None and 'type_to_idx' in ckpt:
            from ecenet import elements
            element_to_type = {elements.symbol(z): idx
                               for z, idx in ckpt['type_to_idx'].items()}
        if element_to_type is None:
            raise ValueError(
                "Checkpoint has no element mapping ('element_to_type' or "
                "'type_to_idx'). Pass element_to_type=... explicitly, or re-save "
                "the checkpoint with a training script (which stores it).")

        # Per-element reference energies. Prefer a ready-made {symbol: eV} dict;
        # otherwise build one from an 'e_ref' array indexed by *this* checkpoint's
        # element mapping (so it is correct for any element set).
        if energy_reference is None:
            energy_reference = ckpt.get('energy_reference')
        if energy_reference is None and 'e_ref' in ckpt:
            e_ref_arr = ckpt['e_ref']
            energy_reference = {sym: float(e_ref_arr[idx])
                                for sym, idx in element_to_type.items()}

        # Units of the model output; default eV. Checkpoints from data in other
        # units (e.g. kcal/mol) store 'energy_units' so the calculator converts.
        if energy_units is None:
            energy_units = ckpt.get('energy_units', 'eV')

        # Mean energy subtracted during training (in the model's units) — add back.
        energy_mean = ckpt.get('energy_mean', 0.0)

        return cls(model, device=device, dtype=dtype,
                   energy_reference=energy_reference,
                   element_to_type=element_to_type,
                   energy_units=energy_units,
                   energy_mean=energy_mean,
                   log_timings=log_timings)

    # ── GPU neighbor list ───────────────────────────────────────────────────

    def _gpu_neighbor_list(self, pos, cell_np, r_cut):
        """O(N²) GPU neighbor list for PBC systems.

        Much faster than ASE's Python implementation for small systems
        (N ≲ 2000). Uses the fractional-coordinate minimum image convention,
        which is exact for orthorhombic cells and a good approximation for
        near-cubic triclinic cells.

        Args:
            pos:      (N, 3) positions, detached GPU tensor
            cell_np:  (3, 3) numpy array, rows = lattice vectors
            r_cut:    cutoff radius in Å

        Returns:
            src, dst:    (n_pairs,) LongTensors  (directed: both i→j and j→i)
            shift_vecs:  (n_pairs, 3) Cartesian PBC shift vectors
        """
        device, dtype = pos.device, pos.dtype
        cell = torch.tensor(cell_np, dtype=dtype, device=device)   # (3, 3)
        inv_cell = torch.linalg.inv(cell)

        # All pairwise raw differences: raw[i, j] = pos[j] - pos[i]
        raw = pos.unsqueeze(0) - pos.unsqueeze(1)                  # (N, N, 3)

        # Minimum image in fractional space
        frac = raw @ inv_cell                                       # (N, N, 3)
        frac = frac - torch.round(frac)
        diff = frac @ cell                                          # (N, N, 3)

        # Filter by cutoff (exclude self-pairs)
        dist2 = (diff ** 2).sum(-1)                                # (N, N)
        mask  = (dist2 < r_cut ** 2) & (dist2 > 1e-20)
        src, dst = mask.nonzero(as_tuple=True)

        # Cartesian shift: forward_pbc uses pos[dst] - pos[src] + shift
        shift_vecs = diff[src, dst] - raw[src, dst]

        return src, dst, shift_vecs

    # ── Core calculation ────────────────────────────────────────────────────

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

    def _t(self):
        self._sync()
        return time.perf_counter()

    def _compute_stress(self, pos, types, edge_i, edge_j, shift_vecs_edge,
                        nb_src, nb_dst, shift_vecs_nb):
        """Strain-based energy / forces / stress for a periodic system.

        Applies an infinitesimal symbolic strain to the positions and the PBC
        shift vectors (``x → x + x·ε``, linear and exact at ε = 0), then takes a
        single backward pass for both forces (−dE/dpos) and dE/dε. The neighbor
        topology is frozen across the strain (standard MLIP approximation).

        Returns ``(energy_tensor, forces_tensor, stress_grad)`` where
        ``stress_grad`` is the (3, 3) dE/dε.
        """
        strain = torch.zeros(3, 3, dtype=self.dtype, device=self.device,
                             requires_grad=True)
        pos_s      = pos + pos @ strain
        shift_e_s  = shift_vecs_edge + shift_vecs_edge @ strain
        shift_nb_s = shift_vecs_nb   + shift_vecs_nb   @ strain
        energy_tensor = self.model.forward_pbc(
            pos_s, types, edge_i, edge_j, shift_e_s, nb_src, nb_dst, shift_nb_s)
        grads = torch.autograd.grad(energy_tensor, [pos_s, strain])
        return energy_tensor, -grads[0], grads[1]

    def _compute_pbc(self, atoms, pos, types, need_stress):
        """Energy / forces (+ optional stress) for a periodic system.

        Builds the edge and neighbour lists under the minimum-image convention,
        then evaluates the model (with strain-based stress if requested).

        Returns ``(energy_tensor, forces_tensor, stress_grad, n_edges, t_nl)``;
        ``stress_grad`` is None when stress was not requested and ``t_nl`` is the
        post-neighbour-list timestamp (None unless ``log_timings``).
        """
        cell = atoms.get_cell().array  # (3, 3), rows = lattice vectors

        # Minimum image convention: cutoff must be <= L/2 in each direction.
        lengths = atoms.cell.lengths()
        max_cut = max(self.model.r_cut_edge, self.model.r_cut_neighbor)
        if not self._mic_warned and any(max_cut > l / 2 for l in lengths):
            import warnings
            warnings.warn(
                f"Cutoff ({max_cut:.2f} Å) exceeds half the box size "
                f"({lengths.min():.2f} Å). Minimum image convention violated. "
                f"Use a larger box (L > {2*max_cut:.1f} Å in all dimensions)."
            )
            self._mic_warned = True

        edge_i, edge_j, shift_vecs_edge = self._gpu_neighbor_list(
            pos.detach(), cell, self.model.r_cut_edge)
        nb_src, nb_dst, shift_vecs_nb = self._gpu_neighbor_list(
            pos.detach(), cell, self.model.r_cut_neighbor)

        t_nl = self._t() if self.log_timings else None

        if need_stress:
            energy_tensor, forces_tensor, stress_grad = self._compute_stress(
                pos, types, edge_i, edge_j, shift_vecs_edge,
                nb_src, nb_dst, shift_vecs_nb)
        else:
            energy_tensor = self.model.forward_pbc(
                pos, types, edge_i, edge_j, shift_vecs_edge,
                nb_src, nb_dst, shift_vecs_nb)
            forces_tensor = -torch.autograd.grad(energy_tensor, pos)[0]
            stress_grad   = None

        return energy_tensor, forces_tensor, stress_grad, len(edge_i), t_nl

    def calculate(self, atoms=None, properties=('energy', 'forces'),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        symbols = atoms.get_chemical_symbols()
        positions_np = atoms.get_positions()  # Å

        # Check all elements are supported
        unsupported = set(s for s in symbols if s not in self.element_to_type)
        if unsupported:
            raise ValueError(f"Unsupported elements: {unsupported}. "
                             f"Supported: {list(self.element_to_type)}")

        types = torch.tensor(
            [self.element_to_type[s] for s in symbols],
            dtype=torch.long, device=self.device
        )
        pos = torch.tensor(
            positions_np, dtype=self.dtype, device=self.device
        ).requires_grad_(True)

        t0 = self._t() if self.log_timings else None

        with torch.enable_grad():
            if atoms.pbc.any():
                (energy_tensor, forces_tensor, stress_grad,
                 n_edges, t1) = self._compute_pbc(
                    atoms, pos, types, 'stress' in properties)
            else:
                n_edges = '—'
                t1 = self._t() if self.log_timings else None
                energy_tensor = self.model.forward(pos, types)
                forces_tensor = -torch.autograd.grad(energy_tensor, pos)[0]
                stress_grad   = None

            t2 = self._t() if self.log_timings else None

        energy = energy_tensor.item() * self._to_ev + self._energy_mean_ev
        forces = forces_tensor.detach().cpu().numpy() * self._to_ev

        if self.log_timings:
            self._step_count += 1
            print(
                f"step {self._step_count:>6d} | "
                f"NL {(t1-t0)*1e3:6.1f} ms | "
                f"fwd {(t2-t1)*1e3:6.1f} ms | "
                f"tot {(t2-t0)*1e3:6.1f} ms | "
                f"edges {n_edges}",
                flush=True
            )

        # Add back per-element reference energies (already in eV)
        for s in symbols:
            energy += self.energy_reference.get(s, 0.0)

        self.results['energy'] = energy
        self.results['forces'] = forces

        if stress_grad is not None:
            volume = abs(np.linalg.det(atoms.get_cell().array))
            stress_mat = stress_grad.detach().cpu().numpy() * self._to_ev / volume
            # ASE Voigt convention: [xx, yy, zz, yz, xz, xy] in eV/Å³
            self.results['stress'] = np.array([
                stress_mat[0, 0], stress_mat[1, 1], stress_mat[2, 2],
                stress_mat[1, 2], stress_mat[0, 2], stress_mat[0, 1],
            ])
