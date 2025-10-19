
# Hybrid LBM + PINN (PyTorch) — Workflow & Results Guide

This guide explains the workflow, how to run the code in 2D (set `nz=1`), what files get written, and how to visualize the results in ParaView.

---

## 1) Workflow at a Glance

**Goal:** Run a lattice Boltzmann method (LBM, D3Q19) in a porous medium and use a physics-informed neural network (PINN) to learn a scalar potential \\( \psi(x,y,z) \\). The correction force \\( \mathbf{F}_\text{corr} = - \nabla \psi \\) is blended into the LBM body force to reduce divergence and momentum residuals.

**Loop:**

1. **LBM chunk** (collision–streaming–bounce-back) for `steps_per_cycle` steps.
2. **Sample** `sample_pts` fluid locations and **collect** velocity + finite-difference gradients.
3. **Train PINN** on GPU (PyTorch) with losses:
   - Divergence: \\( \nabla^2 \psi \approx \nabla \cdot \mathbf{u} \\)
   - Momentum (steady): \\( (\mathbf{u}\cdot\nabla)\mathbf{u} - \nu \nabla^2\mathbf{u} + \nabla\psi \approx \mathbf{0} \\)
   - Weak data consistency: \\( \mathbf{u} + \mathbf{F}_\text{corr} \approx \mathbf{u} \\) (keeps correction gentle)
4. **Infer \\( \mathbf{F}_\text{corr} \\)** on the full grid (batched) and **blend** it into LBM using `corr_relax`:
   \\[ \mathbf{F} \leftarrow \mathbf{F} + \text{corr\_relax}\,\mathbf{F}_\text{corr}. \\]
5. Repeat for `outer_cycles` cycles.

**Notes for 2D runs:** Set `nz = 1`. The code still uses the D3Q19 stencil but the domain has a single z-layer; ParaView will show a flat slab (you can “Slice” or “Extract Surface” if needed).

---

## 2) What Files Are Generated?

At intervals `save_every`, the script writes **VTK legacy** files (ASCII, `STRUCTURED_POINTS`):

- **Solid mask (walls/porous matrix)**: scalar `solid` (1 = solid, 0 = fluid)
- **Density**: scalar `density` (set to 0 in solids for clarity)
- **Velocity**: vector `velocity = (ux, uy, uz)` (0 in solids)

File pattern: `hybrid3d_pt_XXXXXXX.vtk` where `XXXXXXX` is the time-step index (zero-padded).

All fields are written into the **same VTK file** as separate arrays under `POINT_DATA`, so you can toggle visibility in ParaView.

---

## 3) Visualizing in ParaView

1. **Open ParaView** → *File → Open* → select a `hybrid3d_pt_*.vtk` file. When prompted, choose “**Load as a time series**” to animate.
2. Click **Apply**.
3. In the **Properties** panel:
   - For 2D (`nz=1`), use **Slice** (Plane normal = Z) or **Warp by Scalar** on density if you want a height map.
4. In the **Coloring** dropdown (top of the pipeline browser), choose:
   - `solid` (blue = fluid, yellow = solid) or
   - `density` for compressibility variations (should be ~1.0), or
   - `velocity` and then switch to **Glyph** or **Stream Tracer** for vector visualization.
5. To **animate**, press the **Play** button. You can also use **Temporal Statistics** to get mean/max fields over time.

**Tips**
- For 2D look, set **Representation** to “Surface” and apply a **Slice** at z=0.
- Use **Stream Tracer**: Seed source = “Line” crossing the domain; under Display, color by `velocity` magnitude.

---

## 4) Meaning of the Results

- **solid (0/1):** geometry mask. 1 = solid (no-slip), 0 = fluid. Useful to see pore structure.
- **density:** should hover near 1.0 (lattice units) in weakly compressible LBM. Large deviations suggest too strong forcing or numerical issues.
- **velocity:** the flow field evolving through pores. Over cycles, after injecting \\( \mathbf{F}_\text{corr} \\), you should see:
  - Reduction in spurious divergence (improved mass conservation).
  - Momentum residual shrinkage near complex walls.
  - Slight adjustments of the flow distribution; not huge changes (since correction is gently blended).

**Convergence cues**
- Monitor printed `|u|_max` (it should stabilize).
- The PINN’s training losses (`div`, `mom`, `data`) should **decrease** or settle to small values.

---

## 5) Common Pitfalls (and fixes)

- **NaNs in training losses:** reduce `pin_lr` (e.g., `5e-4`), enable gradient clipping (already included), or reduce `sample_pts`. Make sure `nz=1` doesn’t break FD (the code clamps indices so z-derivatives become 0, which is fine).
- **GPU context warnings (Windows):** the script “warms up” CUDA and makes sure tensors are consistently on GPU; if you still see issues, update your NVIDIA driver / CUDA toolkit or run in a clean conda env.
- **ParaView shows a blank screen:** click **Apply** after opening; for `nz=1`, add a **Slice** filter.

---

## 6) Parameters You May Tune

- `corr_relax` (default 0.4): blending factor; smaller = more conservative correction.
- `pin_epochs` / `sample_pts`: training effort per cycle; more samples or epochs = more accurate but slower.
- `tau` (LBM): viscosity via \\( \nu = c_s^2(\tau-0.5) \\). Keep \\( \tau \gtrsim 0.6 \\) for stability.
- `Fx_base`: main driving force; keep small to remain near incompressible regime.

---

## 7) Next Steps (Multiphase)

For multiphase, we will augment the PINN inputs with a phase indicator \\( \phi \\), possibly \\( \nabla\phi \\) and curvature \\( \kappa \\), and add losses for capillary-consistent momentum (CSF), Young–Laplace, and contact angles. The coupling loop remains identical.

---

**Enjoy exploring!**
