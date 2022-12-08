# Diffusion, Denoising, and Deep Networks

A Fall 2022 CS 182 Final Project.

## Introduction

The motivating paper for our problem set is "Denoising Diffusion Probabilistic
Models," which investigates diffusion probabilistic models and denoising with
the goal of generating new high quality image samples.

In these series of Jupyter notebooks, we explore some of the key concepts from
the paper and diffusion models in general.
We hope they can be illustrative!

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

