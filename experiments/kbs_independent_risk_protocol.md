# Independent risk calibration: locked protocol

2026-10-22. Protocol fixed before evaluating the new fitted readouts.

Use existing frozen CLIP L/14 and B/32 feature caches, the existing ten class
partitions and both five-session arrival orders, on CIFAR, Pets and ImageNet-R.
No image encoding or encoder fitting. The original support and test pools remain
unchanged. This is an adaptive benchmark revision: the original test results
have already informed research development, not a newly untouched final test.

## Independent pools

Concatenate original fitting and validation pools. For each class shuffle indices
using RNG seed 662026, then reserve max(1,floor(.4*N)) calibration and
max(1,floor(.2*N)) tuning images; fit uses the remainder (at least two images).
One shared split per dataset for both encoders and every readout/partition.
All current classes have N>=5. Save indices, labels and feature-source hashes.
Only original-class labels enter fitting, tuning and risk calibration. Pseudo
episodes use original base classes and the existing reserved supports. No actual
novel-class query labels enter selection. Ridge/mixtures/cache settings and
temperature scales use fit/tuning only. Refitting is mandatory; the old fitted
readouts cannot be certified using a post-hoc split of their old validation data.

Compare retained tables, recalibrated removal, dense anchored prototypes, raw
fusion, fixed-temperature fusion and class-mean semantic cache with identical
new access. Keep the original grids; cache fitting uses only the new fit pool.

## Protection risk and certification

Let R=1{original reference classifier correct} and C_d=1{expanded readout correct}
on a base-class query. Reference is the newly fitted retained closed-base model.
At retention r=.95, alpha=1-r, define bounded protection loss

L_d = R*(1-C_d) + alpha*(1-R), in [0,1].

For macro population expectation, E L_d <= alpha implies
A_B(d) >= r*A_B,ref. Indeed E L_d = D+alpha*(1-Aref),
A_B(d) = Aref-D+G with recovery G>=0. For retained scores G=0.
The sufficient protection criterion treats all comparators identically, but is
conservative for readouts that recover reference errors. It certifies population
risk under independent, class-conditional identically distributed samples and
fixed score models, not every finite test set or arbitrary distribution shifts.

Calibration weights are w_i=1/(B*n_class); sum_i w_i=1. Set b=max_i w_i.
For observed weighted loss q<alpha use
p=exp(-KL_Bernoulli(q,alpha)/b); otherwise p=1.
This is a valid lower-tail test of E L_d >= alpha for independent bounded
observations, including unequal class-conditional sample counts. Do not use an
ordinary iid binomial interval on pooled, unequally weighted observations.

Proof sketch: for t<0 and mu_i=E L_i, boundedness gives
log E exp(t*w_i*L_i) <= log(1-mu_i+mu_i*exp(t*w_i)).
Convexity in w_i and concavity in mu_i bound the sum by
b^-1 log(1-mu+mu*exp(t*b)), mu=sum_i w_i*mu_i.
Chernoff optimization gives P(q<=x)<=exp(-KL(x,mu)/b) for x<mu.

For each readout/session/order, candidate finite offsets are unique 0..100%
quantiles (101 points) of base tuning margins, ordered from conservative to
permissive. Candidates are fixed without calibration or test labels. Use fixed
sequence testing: advance while p<=delta/5, stop at the FIRST failure, choose
the largest accepted finite offset. This controls selection without treating a
pointwise bound as uniform. Union bound across five sessions gives 1-delta
protection for each prespecified method/order/partition. No simultaneous claim
across methods, orders, datasets, partitions or confidence levels is made.

Prespecified delta values .10,.05,.01 are each separate sensitivity analyses.
Primary delta=.05. Risk sample reuse across sessions is allowed by the union
bound; future support sets must be prespecified independently of calibration
labels, as in these arrival simulations. This is not arbitrary adaptive arrivals.

If no finite offset passes, return the ORIGINAL reference base-only classifier,
with no novel admission. This fallback is structurally safe, separately counted
and has zero novel accuracy. It must not be advertised as useful certified
expansion or as the dense/cache model's own accuracy.

## Empirical controls and reporting

Compare with largest calibration-feasible thresholds at fractions .95,.97,.99
of the same reference calibration macro accuracy, on the SAME newly fitted
models and independent calibration pool. The stricter fractions are diagnostic
controls, not dataset-specific policies chosen using test outcomes.

Report: all six methods, confidence levels, certification/fallback counts,
empirical feasibility, test target attainment, signed and worst shortfall, novel
accuracy and complete five-session trajectory attainment. All partitions count
in operational means with explicit fallback flags; also show jointly nonfallback
paired sets. Ten partitions share fixed images and are not independent datasets.

CPU/GPU resource policy: existing environment, two CPU threads, one local GPU
job, .75 GPU allocation cap, source/input guarded atomic session checkpoints,
shared PAUSE marker. Save compressed sufficient statistics only, no copied
features/models/images. Do not modify earlier source-hashed experiment code.

References: Learn then Test, arXiv:2110.01052, fixed sequence testing. The weighted
protection construction and its exact assumptions must be stated explicitly;
generic conformal expected-risk guarantees do not substitute for this claim.
