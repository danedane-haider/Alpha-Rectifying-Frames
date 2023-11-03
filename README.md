Alpha-rectifying Frames:
# Injectivity and Reconstruction of ReLU-layers

This repository contains routines related to the injectivity of ReLU-layers and $\alpha$-rectifying frames, accompanying our ICML23 paper [Convex Geometry of ReLU-layers, Injectivity on the Ball and Local Reconstruction](http://arxiv.org/abs/2307.09672). It currently includes:

- **polytope bias estimations** (PBE) for different input data domains:
  _given a collection of vectors (e.g. row vectors of a weight matrix), the PBE gives a bias vector_ $\alpha^\mathbb{K}$ _such that any ReLU-layer with bias_ $b\leq\alpha^\mathbb{B}$ _is injective on_ $K$
- **reconstruction formulas** for injective ReLU-layers
