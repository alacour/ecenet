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

    # Whether _energy_pbc consumes the cell tensor. The SR model does not
    # (PBC enters through the shift vectors), so the base skips building it;
    # the LES subclass flips this for its Ewald term.
    _uses_cell = False

    def __init__(self, model, device=None, dtype=torch.float64,
                 energy_reference=None, element_to_type=None,
                 energy_units='eV', energy_mean=0.0,
                 log_timings=False, **kwargs):
        super().__init__(**kwargs)
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
                        energy_units=None, log_timings=False,
                        ignore_les=False, ckpt=None):
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

        # ckpt: optional pre-loaded checkpoint dict — callers that already
        # deserialised the file (load_calculator, the LES subclass) pass it
        # through instead of paying a second multi-hundred-MB torch.load.
        if ckpt is None:
            ckpt = torch.load(checkpoint_path, map_location=device,
                              weights_only=False)

        # Checkpoints trained jointly with LES (train_ecenet_xyz use_les=True)
        # carry a top-level 'les' dict. This calculator computes only the
        # short-range energy, so loading one here would silently drop the
        # long-range term the weights were trained against — refuse instead
        # (same philosophy as the W_msg rejection below). ignore_les=True
        # loads the short-range part deliberately.
        if 'les' in ckpt and not ignore_les:
            raise ValueError(
                "Checkpoint was trained jointly with LES long-range "
                "electrostatics; ECENetCalculator computes only the "
                "short-range energy and would silently drop the E_lr term. "
                "Load through ecenet.calculator.load_calculator (which "
                "dispatches to ECENetLESCalculator for the full E_sr + E_lr), "
                "or pass ignore_les=True to load the short-range part "
                "deliberately."
            )

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
        # Removed features; older checkpoints still carry them in hparams. All
        # were disabled in every training script, so no weights are affected.
        # n_dist_basis only ever fed the removed 'edge' MP layer, so an n_mp=1
        # checkpoint that carries it still loads unchanged.
        for _removed in ('n_dist_embed', 'edge_type_nonlin',
                         'edge_type_linear', 'edge_type_output', 'n_dist_basis'):
            hp.pop(_removed, None)

        state = ckpt.get('best_state') or ckpt.get('model')
        # The distance/type-weighted 'edge' message passing was removed. Its
        # weights (mp_layers.*.W_msg) have no counterpart in the current MP
        # layers, so such a checkpoint would otherwise load with a randomly
        # initialised MP layer and silently return wrong energies.
        if any(k.endswith('W_msg') for k in state):
            raise ValueError(
                "Checkpoint uses the removed mp_type='edge' message passing "
                "(found 'W_msg' in the state dict). That layer no longer exists; "
                "retrain the model with mp_type='softmax' or 'sum'."
            )

        # RealSpaceNonlinearity used to carry fixed pre_scale=1 / pre_shift=0
        # buffers and apply them before the activation — an exact identity, since
        # they were never learnable. Both are gone, so drop them from older
        # checkpoints: they are provably a no-op, and leaving them in would trip
        # the unexpected-key check below.
        state = {k: v for k, v in state.items()
                 if not (k.endswith('.pre_scale') or k.endswith('.pre_shift'))}

        model = ECENet(**hp, n_mp=n_mp)
        if dtype == torch.float64:
            model = model.double()
        model = model.to(device)

        # strict=False tolerates buffers dropped by past refactors, but a real
        # mismatch (missing/unexpected *parameters*) means the architecture
        # rebuilt from hparams disagrees with the weights — surface it instead of
        # silently running a partly random model.
        incompat = model.load_state_dict(state, strict=False)
        _missing = [k for k in incompat.missing_keys if k in dict(model.named_parameters())]
        if _missing or incompat.unexpected_keys:
            raise ValueError(
                "Checkpoint weights do not match the architecture rebuilt from "
                f"'hparams'. Missing parameters: {_missing[:5]}"
                f"{'…' if len(_missing) > 5 else ''}; unexpected keys: "
                f"{incompat.unexpected_keys[:5]}"
                f"{'…' if len(incompat.unexpected_keys) > 5 else ''}."
            )
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

    # ── Energy seams (overridden by ECENetLESCalculator) ────────────────────

    def _energy_free(self, pos, types):
        """Total energy of a non-periodic system (model units)."""
        return self.model.forward(pos, types)

    def _energy_pbc(self, pos, types, edge_i, edge_j, shift_vecs_edge,
                    nb_src, nb_dst, shift_vecs_nb, cell=None):
        """Total energy of a periodic system (model units).

        ``cell`` is the (possibly strained) (3, 3) cell tensor on the autograd
        graph; the short-range model does not use it (PBC enters through the
        shift vectors), but the LES subclass's Ewald term does.
        """
        return self.model.forward_pbc(
            pos, types, edge_i, edge_j, shift_vecs_edge,
            nb_src, nb_dst, shift_vecs_nb)

    def _compute_stress(self, pos, types, edge_i, edge_j, shift_vecs_edge,
                        nb_src, nb_dst, shift_vecs_nb, cell_np): # Prototype, mainly implemented by Claude
        """Strain-based energy / forces / stress for a periodic system.

        Applies an infinitesimal symbolic strain to the positions, the PBC
        shift vectors, and the cell (``x → x + x·ε``, linear and exact at
        ε = 0), then takes a single backward pass for both forces (−dE/dpos)
        and dE/dε. The neighbor topology is frozen across the strain (standard
        MLIP approximation). The strained cell only matters for subclasses
        whose energy uses it (LES Ewald); the short-range model sees the
        strain through positions and shifts.

        Returns ``(energy_tensor, forces_tensor, stress_grad)`` where
        ``stress_grad`` is the (3, 3) dE/dε.
        """
        strain = torch.zeros(3, 3, dtype=self.dtype, device=self.device,
                             requires_grad=True)
        pos_s      = pos + pos @ strain
        shift_e_s  = shift_vecs_edge + shift_vecs_edge @ strain
        shift_nb_s = shift_vecs_nb   + shift_vecs_nb   @ strain
        cell_s = None
        if self._uses_cell:
            cell_t = torch.tensor(cell_np, dtype=self.dtype, device=self.device)
            cell_s = cell_t + cell_t @ strain
        energy_tensor = self._energy_pbc(
            pos_s, types, edge_i, edge_j, shift_e_s, nb_src, nb_dst,
            shift_nb_s, cell=cell_s)
        # allow_unused: a zero-edge structure (every pair beyond r_cut — e.g.
        # a cell relaxed/expanded past dissociation) has a position- and
        # strain-independent energy (Σ atomic offsets), so the leaves never
        # enter the graph; the physical gradient is exactly zero there.
        grads = torch.autograd.grad(energy_tensor, [pos_s, strain],
                                    allow_unused=True)
        f = -grads[0] if grads[0] is not None else torch.zeros_like(pos_s)
        s = grads[1] if grads[1] is not None else torch.zeros_like(strain)
        return energy_tensor, f, s

    def _types(self, symbols):
        """Element symbols → model type indices, rejecting unsupported ones."""
        unsupported = set(s for s in symbols if s not in self.element_to_type)
        if unsupported:
            raise ValueError(f"Unsupported elements: {unsupported}. "
                             f"Supported: {list(self.element_to_type)}")
        return torch.tensor(
            [self.element_to_type[s] for s in symbols],
            dtype=torch.long, device=self.device)

    def _neighbor_lists(self, pos, cell):
        """Edge + neighbour lists for a periodic cell.

        The list flavour is chosen per cell: the fast minimum-image list when
        the cutoff sphere fits inside half the cell's minimum perpendicular
        width (typical MD boxes), and the all-images list — every periodic
        copy within the cutoff, self-image edges included, exactly the
        topology the trainers build — when it does not (small crystal cells:
        ~97% of MPtrj/WBM frames). The two are identical wherever MIC is
        valid.
        """
        from ecenet.radial import min_perpendicular_width, torch_neighbor_list

        max_cut = max(self.model.r_cut_edge, self.model.r_cut_neighbor)
        if max_cut > 0.5 * min_perpendicular_width(cell):
            cell_t = torch.tensor(cell, dtype=self.dtype, device=self.device)
            edge_i, edge_j, shift_vecs_edge = torch_neighbor_list(
                pos.detach(), cell_t, self.model.r_cut_edge)
            nb_src, nb_dst, shift_vecs_nb = torch_neighbor_list(
                pos.detach(), cell_t, self.model.r_cut_neighbor)
        else:
            edge_i, edge_j, shift_vecs_edge = self._gpu_neighbor_list(
                pos.detach(), cell, self.model.r_cut_edge)
            nb_src, nb_dst, shift_vecs_nb = self._gpu_neighbor_list(
                pos.detach(), cell, self.model.r_cut_neighbor)
        return edge_i, edge_j, shift_vecs_edge, nb_src, nb_dst, shift_vecs_nb

    def _compute_pbc(self, atoms, pos, types, need_stress):
        """Energy / forces (+ optional stress) for a periodic system.

        Builds the edge and neighbour lists (see ``_neighbor_lists``), then
        evaluates the model (with strain-based stress if requested).

        Returns ``(energy_tensor, forces_tensor, stress_grad, n_edges, t_nl)``;
        ``stress_grad`` is None when stress was not requested and ``t_nl`` is the
        post-neighbour-list timestamp (None unless ``log_timings``).
        """
        cell = atoms.get_cell().array  # (3, 3), rows = lattice vectors
        (edge_i, edge_j, shift_vecs_edge,
         nb_src, nb_dst, shift_vecs_nb) = self._neighbor_lists(pos, cell)

        t_nl = self._t() if self.log_timings else None

        if need_stress:
            energy_tensor, forces_tensor, stress_grad = self._compute_stress(
                pos, types, edge_i, edge_j, shift_vecs_edge,
                nb_src, nb_dst, shift_vecs_nb, cell)
        else:
            cell_t = (torch.tensor(cell, dtype=self.dtype, device=self.device)
                      if self._uses_cell else None)
            energy_tensor = self._energy_pbc(
                pos, types, edge_i, edge_j, shift_vecs_edge,
                nb_src, nb_dst, shift_vecs_nb, cell=cell_t)
            g = torch.autograd.grad(energy_tensor, pos, allow_unused=True)[0]
            forces_tensor = -g if g is not None else torch.zeros_like(pos)
            stress_grad   = None

        return energy_tensor, forces_tensor, stress_grad, len(edge_i), t_nl

    def calculate(self, atoms=None, properties=('energy', 'forces'),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        symbols = atoms.get_chemical_symbols()
        positions_np = atoms.get_positions()  # Å
        types = self._types(symbols)
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
                energy_tensor = self._energy_free(pos, types)
                # allow_unused: zero-edge systems (dissociated fragments)
                # have a position-independent energy; forces are exactly 0.
                g = torch.autograd.grad(energy_tensor, pos,
                                        allow_unused=True)[0]
                forces_tensor = -g if g is not None else torch.zeros_like(pos)
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


class ECENetLESCalculator(ECENetCalculator):
    """ASE calculator for checkpoints trained jointly with LES.

    Evaluates ``E = E_sr + E_lr`` on one autograd graph, so forces and stress
    need no extra code: forces come from the joint backward, and the strain
    pass strains positions, shift vectors, AND the cell, covering the Ewald
    term's explicit cell dependence (exactly the xyz trainer's convention).
    Non-periodic systems use the isolated pairwise path (``cell=None``);
    periodic ones the reciprocal-space Ewald.

    Construct via ``from_checkpoint`` (a checkpoint with a top-level ``les``
    dict — this class refuses short-range checkpoints, symmetrically with
    ECENetCalculator refusing LES ones). The upstream charge head is
    materialised and its trained state loaded here; edge-mode read-outs
    (``les_readout='edge'/'edge_basis'``) carry the charge (and with
    ``les_dipole`` the bond dipoles, with ``les_alpha`` the latent
    polarizabilities) inside the model itself, so the LES module is then
    parameter-free.
    """

    _uses_cell = True
    implemented_properties = ['energy', 'forces', 'stress', 'charges']

    def __init__(self, model, les_module=None, **kwargs):
        """The l0 convention (l0_is_charge / les_dipole) is read off the
        model — it is a pure function of the model's ``les_readout`` and
        ``les_dipole`` hparams, so taking it as arguments could only ever
        mis-pair them (silently: a wrong ``l0_is_charge`` would route l0
        through a randomly-initialised upstream head). ``les_module=None``
        exists so the base ``from_checkpoint`` can construct the instance
        before the module is built; it is attached right after.
        """
        super().__init__(model, **kwargs)
        self.les_module = les_module
        self.les_flags = model.les_flags

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None, dtype=None,
                        energy_reference=None, element_to_type=None,
                        energy_units=None, log_timings=False, ckpt=None):
        """Load model + LES module from a joint checkpoint (see base class)."""
        if ckpt is None:
            ckpt = torch.load(checkpoint_path, map_location='cpu',
                              weights_only=False)
        if 'les' not in ckpt:
            raise ValueError(
                "Checkpoint carries no 'les' dict — it is a short-range "
                "model; use ECENetCalculator.from_checkpoint instead.")

        calc = super().from_checkpoint(
            checkpoint_path, device=device, dtype=dtype,
            energy_reference=energy_reference, element_to_type=element_to_type,
            energy_units=energy_units, log_timings=log_timings,
            ignore_les=True, ckpt=ckpt)

        from ecenet.les import load_les_module
        calc.les_module = load_les_module(ckpt['les'], calc.model,
                                          calc.device, calc.dtype)
        return calc

    # ── Energy seams: add E_lr on the same graph ────────────────────────────
    #
    # Both seams also stash the per-atom latent charges (and, with
    # ``les_dipole`` / ``les_alpha``, the latent atomic dipoles and
    # polarizabilities) into ``self.results`` as a side effect — every force
    # call computes them anyway, so exposing them is free.
    # ``atoms.get_charges()`` reads ``results['charges']`` (e);
    # ``results['les_dipoles']`` holds u (N, 3) in e·Å; ``results['les_alphas']``
    # holds α^les as (N,) or (N, 3, 3) in e·Å/(V/Å). Stashing is safe in
    # the strain (stress) pass too: it evaluates at ε = 0, i.e. the
    # unstrained geometry, so it stores the same numbers. The global sign of
    # the latent charges is arbitrary (E_lr is quadratic in q) — consistent
    # within a checkpoint, not physically pinned.

    def _stash_charges(self, q, l0):
        self.results['charges'] = q.detach().cpu().numpy().reshape(-1)
        if self.model.les_dipole or self.model.les_alpha:
            from ecenet.les import unpack_l0
            _, u, alpha = unpack_l0(l0, **self.les_flags)
            if u is not None:
                self.results['les_dipoles'] = u.detach().cpu().numpy()
            if alpha is not None:
                self.results['les_alphas'] = alpha.detach().cpu().numpy()

    def _energy_free(self, pos, types):
        e_sr, l0 = self.model.forward(pos, types, return_embeddings=True,
                                      l0_only=True)
        e_lr, q = self.les_module(l0, pos, return_charges=True,
                                  **self.les_flags)
        self._stash_charges(q, l0)
        return e_sr + e_lr.sum()

    def _energy_pbc(self, pos, types, edge_i, edge_j, shift_vecs_edge,
                    nb_src, nb_dst, shift_vecs_nb, cell=None):
        e_sr, l0 = self.model.forward_pbc(
            pos, types, edge_i, edge_j, shift_vecs_edge,
            nb_src, nb_dst, shift_vecs_nb,
            return_embeddings=True, l0_only=True)
        e_lr, q = self.les_module(l0, pos, cell=cell, return_charges=True,
                                  **self.les_flags)
        self._stash_charges(q, l0)
        return e_sr + e_lr.sum()

    def compute_bec(self, atoms):
        """Born effective charges Z* = ∂P/∂r for one structure, (N, 3, 3).

        Puts the positions on the autograd graph, so the latent charges q(r)
        (and dipoles u(r)) stay functions of the positions and upstream's BEC
        module delivers the charge-flow terms Σⱼ rⱼ ∂qⱼ/∂rᵢ — the part a
        static-charge picture misses. Upstream handles the polarization
        (Berry-phase-style for periodic cells, direct sum otherwise),
        mean-charge removal, and the √ε∞ normalisation as configured in the
        checkpoint's ``les_arguments``; with ``les_dipole`` the charge and
        dipole parts are summed, and with ``les_alpha`` the induced dipoles
        Δu_i = α_i·E_i are part of the polarization (see
        ``LESLongRange.born_charges``).

        Unlike the latent-charge stash this cannot ride along ``calculate()``
        — the force backward frees the graph, and Z* needs gradients of the
        three polarization components — so it runs its own forward plus three
        backward passes (≈ 4 force calls' cost). Topology is frozen from the
        detached positions (standard). Same arbitrary global sign as the
        latent charges. Note the periodic path calls ``torch.linalg.det/inv``
        on the cell — the float32-CUDA det bug on cu13 torch applies.
        """
        symbols = atoms.get_chemical_symbols()
        types = self._types(symbols)
        pos = torch.tensor(atoms.get_positions(), dtype=self.dtype,
                           device=self.device).requires_grad_(True)
        with torch.enable_grad():
            if atoms.pbc.any():
                cell_np = atoms.get_cell().array
                (edge_i, edge_j, she,
                 nb_src, nb_dst, shn) = self._neighbor_lists(pos, cell_np)
                _, l0 = self.model.forward_pbc(
                    pos, types, edge_i, edge_j, she, nb_src, nb_dst, shn,
                    return_embeddings=True, l0_only=True)
                cell_t = torch.tensor(cell_np, dtype=self.dtype,
                                      device=self.device).view(-1, 3, 3)
            else:
                _, l0 = self.model.forward(pos, types, return_embeddings=True,
                                           l0_only=True)
                cell_t = None
            bec = self.les_module.born_charges(l0, pos, cell=cell_t,
                                               **self.les_flags)
        return bec.detach().cpu().numpy()

    def compute_polarizability(self, atoms):
        """Polarizability tensor α = Σ_i α_i for one structure, (3, 3).

        Needs a ``les_alpha`` checkpoint. The induced dipoles respond
        linearly and non-self-consistently to the field, and the fixed
        multipoles don't respond to an external field at all, so the
        polarization response is exactly the sum of the atomic tensors
        (paper Eq. 29) — one forward, no gradients. An isotropic ``'iso'``
        model gives a multiple of the identity. Units e·Å/(V/Å) = e·Å²/V
        (multiply by 1/(4πε₀) = 14.3996 eV·Å/e² for Å³). This is the latent
        α^les: physical for an isolated molecule (ε_e = 1); for bulk it needs
        the ε_e = ε_∞/(1 + χ^les) unscaling of the paper's Eqs. 23–24, which
        needs ε_∞ and is left to the caller. Unlike the latent charges the
        sign is physical: E_lr is linear in α.
        """
        if self.model.les_alpha is None:
            raise ValueError("compute_polarizability needs a checkpoint "
                             "trained with les_alpha ('iso' or 'aniso')")
        from ecenet.les import total_polarizability, unpack_l0
        symbols = atoms.get_chemical_symbols()
        types = self._types(symbols)
        pos = torch.tensor(atoms.get_positions(), dtype=self.dtype,
                           device=self.device)
        with torch.no_grad():
            if atoms.pbc.any():
                cell_np = atoms.get_cell().array
                (edge_i, edge_j, she,
                 nb_src, nb_dst, shn) = self._neighbor_lists(pos, cell_np)
                _, l0 = self.model.forward_pbc(
                    pos, types, edge_i, edge_j, she, nb_src, nb_dst, shn,
                    return_embeddings=True, l0_only=True)
            else:
                _, l0 = self.model.forward(pos, types, return_embeddings=True,
                                           l0_only=True)
            _, _, alpha = unpack_l0(l0, **self.les_flags)
            pol = total_polarizability(alpha)[0]
        return pol.cpu().numpy()


def load_calculator(checkpoint_path, verbose=True, **kwargs):
    """Load the right calculator for a checkpoint, whatever it was trained with.

    One deserialisation, one dispatch: joint-LES checkpoints (top-level
    ``les`` dict) get :class:`ECENetLESCalculator` so MD and single points
    run on the trained PES; short-range ones get :class:`ECENetCalculator`.
    Every driver/example should load through here rather than re-implementing
    the peek; ``verbose`` announces the LES dispatch, so callers need not
    either. ``kwargs`` are forwarded to ``from_checkpoint``.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    use_les = 'les' in ckpt
    if verbose and use_les:
        print("[les] joint-LES checkpoint — using ECENetLESCalculator "
              "(E_sr + E_lr)")
    cls = ECENetLESCalculator if use_les else ECENetCalculator
    return cls.from_checkpoint(checkpoint_path, ckpt=ckpt, **kwargs)
