# FIMS: Data Generation Code
## Table of Contents:
1. electronAvalanche - Garfield++ simulation of the avalanche process for a single electron within a uniform electric field to determine the gain.
2. FIMS_Geometry - Finite element calculations of FIMS geometry. Geometry is defined using Gmsh and electric fields then calculated using Elmer.
3. FIMS_SIM - Generation and simulation of the FIMS geometry. Utilizes Gmesh and Elmer to define the geometry and the calculate a finite element map of electric fields. Then performs Garfield++ simulations of single-electron avalanches.
