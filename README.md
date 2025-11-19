# Sigma-Aware TBPSA — Solving Nevergrad Issue #1700

Welcome to our GitHub page, where we try to solve issue **#1700** of the original Nevergrad repo.  
**Link to issue #1700:** https://github.com/facebookresearch/nevergrad/issues/1700

---

## Problem Description

The user has this problem : they have a noisy objective function.  
When they evaluate the function at some point $x$, they don't get a single value $y$, but  
**$y \pm \sigma$** (the noise level). 

So for every observation, the user knows how much noise there is in the measurement.

Another important point is that $\sigma$ is not constant across the search space.  
Issue #1700 states *"is typically larger when far away from the optimum"*.

---

## What is TBPSA?

In this problem, the user uses **TBPSA**.  
TBPSA stands for **Test-Based Population Size Adaptation**.  
It is an optimizer in Nevergrad.

It is an optimization method that works well for continuous and noisy cases.  
It adjusts the population size based on statistical tests and recent performance of the evaluations.

---

## How TBPSA Works

1) Start with a population (set of candidates around the current best point)  
2) Evaluate each candidate : compute the noisy function  
3) Compare candidates using statistical tests  
4) Update the current best point with the best candidate  
5) Adapt population size :  
   - if the improvement is uncertain or noisy : **TBPSA increases the population size** to gather more evidence  
   - if the improvement is clear and strong : **TBPSA reduces the population size** to save evaluations  
6) repeat

**TBPSA documentation:**  
https://www.lamsade.dauphine.fr/~cazenave/papers/games_cec.pdf

---

## Why Sigma Matters

So, using TBPSA, we must use **$\sigma$ (the noise information we have about each observation $y$)**  
to determine whether or not we should choose this point when searching for the minimum of a function.

This will help with **stability** and **efficiency** when minimum-searching.

The user suggests an idea :

> "A naive approach I could see is to [...] 'tell' the same datapoint multiple times if the associated error is small"  
→ evaluate the same point multiple times if $\sigma$ is small.

---

## What This Project Does

In this project, we compare:

### Regular TBPSA  
- does **not** use the noise information  
- every candidate is evaluated once  

### Sigma-weighted TBPSA  
- extracts $\sigma(x)$ from the noisy function  
- evaluates the same point **multiple times based on $\sigma(x)$**  
- **averages** results to reduce noise  
- uses this estimate to update TBPSA

---

## Noisy Function Used

We use a heteroskedastic quadratic function:

### True function
$$
f(x) = \sum_i x_i^2
$$

### Noisy observation
$$
y(x) = \mathcal{N}(0, \sigma(x)^2)
$$

### Noise model
$$
\sigma(x) = \sigma_{\min} + (\sigma_{\max} - \sigma_{\min}) \tanh(\alpha \lVert x \rVert)
$$

Properties:
- **near the optimum:** low noise  
- **far away:** high noise  

---

## Takeaways of This Repo

- how to use known noise values for better optimization  
- using TBPSA when noise is not the same across the search space  
- how to optimize a noisy function  
