<p align="center">
  <img src="https://raw.githubusercontent.com/khoomeik/SATScale/main/satscale.png" height="300" alt="Scaling Boolean Satisfiability" />
</p>
<p align="center">
  <em>Towards scalable solutions for boolean satisfiability</em>
</p>
<p align="center">
  <a href="https://twitter.com/khoomeik">🐦 Twitter</a>
</p>

# SATScale
Solving boolean satisfiability in polynomial time proves P = NP. Yet, state-of-the-art neural SAT solvers have <1M parameters and the largest dataset has 120k examples. Indeed, neural approaches to SAT remain far behind their classical/heuristic counterparts.

Maybe we haven't thrown enough GPUs at SAT? And maybe post hoc interpretability work can extract novel discrete algorithms for SAT?

I wonder whether the dominance of tough-to-scale (but mathematically intuitive) GNN approaches to neural SAT is akin to syntax-based (but linguistically intuitive) approaches to language modeling prior to the transformer.

I also notice this correspondence between SAT <-> transformer + binary classifier:
- variable count <-> embedding dimensionality
- clause <-> token
- clause count <-> sequence length
- satisfiability <-> classification

In this setting, we don't need position embeddings since clause conjunction is order-invariant. Each clause embedding is a variable-count-dimensional sparse vector with (n-SAT) n elements either 1 or -1 (negated).

# Data
My experiments so far have been with [Uniform Random-3-SAT from SATLIB](https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html), but this scaling approach will definitely fail with such a small dataset. I hope to soon write a script to generate millions of SAT training examples of varying sizes & properties.
