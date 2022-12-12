# Diffusion, Denoising, and Deep Networks

A Fall 2022 CS 182/282A Final Project.

## Introduction

The motivating paper for our problem set is "Denoising Diffusion Probabilistic
Models," which investigates diffusion probabilistic models and denoising with
the goal of generating new high quality image samples.

In these series of written homework problems and Jupyter notebooks,
we explore some of the key concepts from the paper and diffusion models
in general. We hope the written and coding problems can be illustrative!

## Setup

Before you begin, you'll need to set up your environment.

### If you're running locally:

To set up a new environment that contains the necessary libraries,
you can run the startup script by running `bash startup.sh`.

Double check that you are now in the `env-proj` conda environment that was created.
If not, run `conda activate env-proj` in your terminal.

### If you're running on Colab

Please change the following cell from `%matplotlib widget` to `%matplotlib inline`.
since Colab does not currently seem to support `ipympl` or `widget` out of the box.
(It may be able to support the `widget` mode but more setup for that may be required.)

## References

Here are the main references that we found helpful in creating our problem set.
The links are to their respective GitHub repositories.

\[[1](https://github.com/hojonathanho/diffusion)\]
Jonathan Ho, Ajay Jain, Pieter Abbeel; *Denoising Diffusion Probabilistic Models*;
Advances in Neural Information Processing Systems 33 (NeurIPS) (2020).

\[[2](https://github.com/google/jax)\]
The JAX authors, Google; JAX reference documentation;
https://jax.readthedocs.io/en/latest/index.html (2020).
