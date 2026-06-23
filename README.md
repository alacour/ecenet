# ECENet

**ECENet** is an SO(3)-equivariant interatomic potential (MLIP). It uses
per-edge, SO(2)-equivariant features and passes SO(3)-equivariant messages through the nodes. The use of SO(2) features enables faster, more expressive operations than is possible with SO(3) features.

- Exactly SO(3)-invariant energies (verified to ~1e-11 on random rotations).
- Equivariant message passing (`n_mp >= 2`).
- Energy / forces / stress via autograd; ASE calculator for MD and relaxations.

> 📄 **Paper forthcoming.** A preprint describing the method is in preparation.

## Layout

```
ecenet/
  model.py         ECENet — the model (message passing when n_mp >= 2)
  equivariant.py   EquivariantLinear, RealSpaceNonlinearity
  ace_basis.py     analytic ACE basis + Wigner-rotation autograd functions
  spherical.py     real spherical harmonics, Clebsch–Gordan, Wigner-D (recursion + rotation)
  radial.py        radial bases, cutoff envelopes, edge/neighbour lists
  calculator.py    ECENetCalculator (ASE)
  datasets/mptrj.py  MPtrj dataset loader

scripts/               training / data entry points (run from the repo root)
  train_ecenet.py        rMD17 / MD22 single-molecule training
  train_ecenet_spice.py  SPICE multi-molecule training (10 elements, DDP)
  train_ecenet_mptrj.py  MPtrj training (periodic crystals, ~89 elements, stress)
  prepare_mptrj.py       tensorise raw MPtrj JSON → .pt shards
  eval_spice.py          evaluate a SPICE checkpoint on the test set

examples/              runnable examples + a small example checkpoint
  run_md_*.py            ASE MD drivers (NVT/NPT); importable or CLI
  ethanol.mdl            example rMD17 ethanol model (used by the tests)

tests/                 test suite (test_*.py, run from the repo root)
tools/                 developer utilities (run from the repo root)
  profile_step.py        profile a single calculator step
  equiv_vs_ref.py        numerical-equivalence check vs a git ref
```

## Install

ECENet needs PyTorch, NumPy, ASE, and `sphericart-torch` (for spherical
harmonics). Install it as a package (editable, so `import ecenet` resolves from
anywhere):

```bash
conda create -n ecenet python=3.11
conda activate ecenet
pip install -e .
```

On a GPU machine, install the torch wheel that matches your CUDA version first
(see <https://pytorch.org/get-started/locally/>), then `pip install -e .`.

ECENet is pure PyTorch — no compiled/custom CUDA extensions to build.

**Tested with:** Python 3.11 + CUDA PyTorch (cluster) and Python 3.14 + PyTorch
2.10 CPU (local); NumPy 2.4, ASE 3.28. The dependency floors in `pyproject.toml`
are deliberately conservative.

## Quickstart

Run everything from the repo root so `import ecenet` resolves. All three trainers
are **import-and-call** — every option is a keyword argument of the training
function. The multi-GPU trainers (`train_ecenet_spice`, `train_ecenet_mptrj`)
additionally keep a `__main__` entry point so they launch directly under
`torchrun`; set hyperparameters in the call at the bottom of the script (or
import the function from your own driver).

Train on an rMD17 / MD22 molecule (import-and-call):

```python
from scripts.train_ecenet import train_ecenet
model, results = train_ecenet(molecule='ethanol', n_train=950,
                              l_max=3, n_max=4, embed_dim=16, n_epochs=200,
                              n_mp=2)            # n_mp ≥ 2 turns on message passing
```

Train on SPICE dataset (10 elements):

```python
from scripts.train_ecenet_spice import train_ecenet_spice
model, results = train_ecenet_spice(l_max=3, n_max=4, embed_dim=32, n_layers=2)
```

Multi-GPU via `torchrun`
(`LOCAL_RANK`/`RANK`/`WORLD_SIZE` are read from the environment for DDP):

```bash
python scripts/train_ecenet_spice.py            # single process

torchrun --nproc_per_node=4 scripts/train_ecenet_spice.py    # 4-GPU DDP
```

Use a trained model from Python / ASE:

```python
from ase.io import read
from ecenet.calculator import ECENetCalculator

atoms = read('molecule.xyz')
atoms.calc = ECENetCalculator.from_checkpoint('model.mdl')
print(atoms.get_potential_energy())   # eV
print(atoms.get_forces())             # eV/Å
print(atoms.get_stress())             # eV/Å³ (periodic systems)
```

```python
import ecenet
model = ecenet.ECENet(n_types=10, l_max=3, n_max=4, embed_dim=16)
energy = model(positions, types)      # positions (N,3), types (N,)
```

## Tests

The test suite is pure PyTorch and runs on CPU. Each file is runnable as a script:

```bash
python tests/test_ecenet.py                  # ECENet integration: SO(3) invariance, forces, MP
python tests/test_mptrj_trainer.py           # end-to-end MPtrj trainer smoke (synthetic)
```

## License

Copyright ©2026. The Regents of the University of California (Regents). All
Rights Reserved. Permission to use, copy, modify, and distribute this software
and its documentation is hereby granted, provided that the above copyright
notice, this paragraph and the following two paragraphs appear in all copies,
modifications, and distributions.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE
USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS
PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

See [`LICENSE`](LICENSE) for the full text.
