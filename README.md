# Alpha-rectifying Frames: Injectivity and Reconstruction of ReLU-layers

This repository contains routines related to $\alpha$-rectifying frames and the injectivity of ReLU-layers, including

- **polytope bias estimations** (PBE) for different input data domains (currently only the ball $\mathbb{B}_r$):
  _given a collection of vectors (e.g. row vectors of a weight matrix), the PBE gives a bias vector_ $\alpha^\mathbb{B}$ _such that any ReLU-layer with bias_ $b\leq\alpha^\mathbb{B}$ _is injective on_ $\mathbb{B}$
- **reconstruction formulas** for injective ReLU-layers

The routines make use of the software Polymake 4.6 to compute the facet-incidence matrix of a convex polytope on the sphere. Installing Polymake is flawless, see https://polymake.org/doku.php/download/start. 
