# ACPR

**Anisotropic Covariance-Based Point Cloud Registration for Forests and Wetlands**

This repository contains selected computational components of ACPR: displacement-based hypothesis selection, local covariance models, anisotropic residuals, robust losses, and pose derivatives. It is a partial source release and does not provide an end-to-end registration system or reproduce the paper's experiments.

## Included components

| File | Contents |
| --- | --- |
| [core/modal_initialization.py](core/modal_initialization.py) | Source moments, point-displacement distances between supplied hypotheses, and compact-kernel mode selection. |
| [core/local_covariance.py](core/local_covariance.py) | Local PCA on supplied neighborhoods, planar reliability checks, and regularized covariance models. |
| [core/anisotropic.py](core/anisotropic.py) | Rotation of source covariances, Mahalanobis residuals, and the covariance contribution to the rotation gradient. |
| [core/residual_calibration.py](core/residual_calibration.py) | Initial residual-scale estimation, cutoff calibration, Huber and Tukey losses, and radial weights. |
| [core/rigid_motion.py](core/rigid_motion.py) | Skew matrices, centered residual Jacobians, and application of a supplied SE(3) increment. |
| [core/linearization.py](core/linearization.py) | Objective value, exact local gradient, approximate metric, and weight diagnostics at a supplied pose. |

The covariance, residual, robust-loss, and motion routines have been separated from the experimental implementation and given explicit function arguments. The modal-selection module is a NumPy implementation of the manuscript's displacement-distance and kernel-mode formulas. Input validation and docstrings describe the standalone interfaces. Parameter values used in the experiments are not bundled with these routines.

## Hypothesis selection and geometry

For source endpoints with centroid $\mu$ and population second moment $C$, the distance between two hypotheses is

$$d(T_a,T_b)^2 = \frac{\|(R_a-R_b)\mu+t_a-t_b\|^2+\operatorname{tr}[(R_a-R_b)C(R_a-R_b)^\top]}{s_b^2}.$$

This equals the mean squared displacement of the source points divided by the squared branch support. `hypothesis_distances` returns its square root. `select_modal_hypothesis` sums the compact kernel $[1-(d/h)^2]^2$ for $d<h$ and returns the index of an existing hypothesis with maximum support. Hypotheses must already be available; the module does not generate them or refit their consensus. The implementation evaluates one score at a time instead of allocating a complete pairwise-distance matrix.

`covariances_from_neighborhoods` accepts an array of neighborhoods already selected by the caller. Local PCA gives the surface variation and, when reliable, a normal $n_i$. The covariance model is

$$C_i=\kappa_C[I-(1-\epsilon)n_i n_i^\top].$$

Unreliable neighborhoods use $\kappa_C I$. Neighborhood search, correspondence selection, and stage-specific parameter choices are outside the release.

## Mathematical conventions

Source and target arrays contain matched points as rows. A pose acts as

$$x_i = R p_i + t,$$

which is written as `source @ rotation.T + translation` in NumPy. Hypothesis selection uses physical coordinates and the branch support $s_b$. The anisotropic objective uses coordinates and translation divided by the support $s$, together with dimensionless covariance models. Their physical covariance matrices are $s^2 C_i$. Covariance inputs must be symmetric positive definite, and rotations must be proper rotations.

The source covariance rotates with the pose:

$$A_i = R C_{p_i} R^\top,\qquad \Sigma_i = A_i + C_{q_i}.$$

With $e_i = x_i - q_i$ and $z_i = \Sigma_i^{-1} e_i$, the residual magnitude is $u_i = \sqrt{e_i^\top z_i}$. For a left rotation increment with generator $G_k$, the covariance contribution to the gradient of $\tfrac12 u_i^2$ is

$$-\frac12 z_i^\top (G_k A_i - A_i G_k) z_i.$$

`covariance_rotation_gradient` returns this contribution for each correspondence. `evaluate_objective` combines it with the residual-Jacobian term, radial weights, and averaging to obtain the full local gradient. The objective is a mean radial loss; it has no log-determinant term.

Scale calibration uses the residuals at the initial anisotropic estimate:

$$\sigma_0 = \frac{\operatorname{median}(u_i)}{m_\chi},\qquad b = c_T\sigma_0.$$

Here, `chi_median` is the median of the chi distribution for the residual dimension, and `tukey_multiplier` is $c_T$. Both are supplied by the caller. The correspondence set and calibrated cutoff remain fixed within the subsequent refinement stage. An invalid or effectively zero scale raises `ValueError`; retaining the incoming pose is handled by the omitted registration pipeline.

## Pose derivatives

The pivot $a$ is the centroid of the transformed active source points at the start of a stage. With a left increment ordered as rotation followed by translation, the residual Jacobian is

$$J_i=[-[x_i-a]_\times\;\;I].$$

`left_increment` applies a caller-supplied increment about this pivot using the SE(3) exponential. It does not compute or accept an optimization step. The pivot stays fixed when comparing objective evaluations within a stage.

`evaluate_objective` returns an `ObjectiveEvaluation` with:

- The mean loss and six-component gradient, including covariance rotation.
- The approximate metric $H=N^{-1}\sum_i w_i J_i^\top\Sigma_i^{-1}J_i$.
- Mahalanobis distances, radial weights, and the mean covariance-gradient contribution.
- The number of positive weights, their sum, and effective sample count $(\sum_i w_i)^2/\sum_i w_i^2$.

The effective sample count is zero when all weights are zero. The metric is positive semidefinite and can be singular; it is not the exact Hessian. These diagnostics do not implement the method's geometric validity or acceptance decisions.

## Release scope

The repository does not include feature extraction or matching, compatibility-graph construction, transformation-hypothesis generation, consensus refitting, the Euclidean warm start, neighborhood search, active-set preparation, branch selection, the iterative optimizer, damping and line-search policy, acceptance checks, or dense refinement. It also excludes dataset loaders, experimental parameter configurations, benchmarks, data, and model outputs.

The files provide numerical routines only. There is no command-line entry point, registration demo, or complete solver. Supplying two raw point clouds is not sufficient to run ACPR with this release.

## Dependency

The included routines use NumPy and the Python standard library. Import them as modules under `core` from the repository root. There are no import-time experiments, file writes, downloads, or external registration calls.
