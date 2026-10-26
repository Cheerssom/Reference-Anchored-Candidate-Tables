# What the effective-bias identity does and does not imply

For the retained readout, ignoring a text-centering term shared by every candidate, let

    B_b = q_T,b / sigma_T + a q_L,b / sigma_L
    N_n = q_T,n / sigma_T + u q_P,n / sigma_P.

The original centered decision is equivalent in exact arithmetic to selecting novel when

    max_b B_b - max_n N_n < d + Delta(x),
    Delta(x) = a mu_L(x)/sigma_L(x) - u mu_P(x)/sigma_P(x),

with the original canonical tie rule. Query-dependent scales still affect within-group fusion; centering contributes the cross-group offset. Fixed nonzero means combined with query-dependent scales also produce a query-dependent offset. The identity is a decomposition, not an optimality claim or a guarantee that Delta alone separates base from novel.

An additional constraint follows from the actual inverse-frequency weighted ridge readout with an unregularized intercept. Write its design matrix X=[features,1], weight matrix Omega, positive slope regularizer L with L e_intercept=0, and one-hot targets Y with Y1=1. The normal equations give

    (X' Omega X + L) W1 = X' Omega 1
                           = (X' Omega X + L) e_intercept.

For the unique regularized solution, W1=e_intercept. Thus the learned scores sum to one for every query, and mu_L(x)=1/B in exact arithmetic. The learned contribution to Delta varies through sigma_L(x), not a freely varying learned-score mean. The prototype mean is v' mean_b p_b and can vary with the query. Floating-point ridge solves introduce numerical residuals. A synthetic imbalanced-class test checks this identity; it is not a benchmark performance result or a claim to have individually rechecked all historical weight matrices.

For old-class queries, adding novel rows without changing existing scores leaves the within-base winner fixed. A reference-incorrect sample cannot become correct solely through novel admission. Therefore retained G=0 and destruction-to-other-base=0. For replacement readouts, D contains both destruction to novel and destruction to another base class, and G may be positive. All use the same A_ref. Role separation and this algebra do not establish independent sampling or universal deployment protection.

The new diagnostics are post-hoc on existing benchmarks. Their conditional intervals do not represent independent datasets or broad application populations. The preceding manuscript's fresh-agent audit does not automatically extend to this new analysis.
