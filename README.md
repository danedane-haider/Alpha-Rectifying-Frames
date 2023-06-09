# Alpha-rectifying frames

This repository contains routines related to $\alpha$-rectifying frames and ReLU-layers, including

- polytope bias estimations (PBE) for different input data domains (currently only the ball $\mathbb{B}_r$)
- reconstruction formulas for injective ReLU-layers

The routines make use of the software Polymake to compute the facet-incidence matrices of a convex polytope, given by its vertices on the sphere. Installing Polymake is flawless, see https://polymake.org/doku.php/download/start. 
