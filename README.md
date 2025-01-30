# Examples of using PyTorch

Author: @adills

## Setup
I use `pipenv` for environment setup as defined the the Pipfile.

I'll update later with specifics.

## General neural network 
(add info here)

## PINN
Physics Informed Neutral Networks, and many varieties there in, have gaining traction over the past ten years in science and engineering disciplines due to their ability to model ordinary differential equations and partial differentiation equations, to include system of these equations. 

my examples are inspired from various examples that I've found in academic literature, GitHub repositories, and Medium articles.  my documentation is limited but I try to property reference the materials I used within the code itself. 

I'm currently working an example using a system of linear ordinary differential (ODE) equations similar to a couples spring mass system. I started with a basic PINN that determines the displacements that over the ODEs. and I'm working a variety of extensions from there.  The with is being done in the `sysEqns.ipynb` notebook and later will be incorporated into one or more `.py` files.