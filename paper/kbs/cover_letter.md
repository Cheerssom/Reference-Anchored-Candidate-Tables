Dear Editors of Knowledge-Based Systems,

Please consider our manuscript, "Reference-anchored asymmetric class expansion with query-adaptive evidence alignment," as an original Research Article.

The paper addresses class expansion when a trained base classifier and arriving few-shot classes provide asymmetric evidence. Fixed base references align learned scores, semantic anchors and visual prototypes while preserving existing scores during vocabulary growth. Query normalization induces an explicit sample-adaptive admission margin without an auxiliary routing network. Independent calibration selects the global offset under a specified base-accuracy requirement, enabling retention-constrained expansion and frozen-model transfer.

Experiments cover three source datasets, ten class partitions, two frozen CLIP encoders and five arrivals. In matched transfers on ImageNet-R, average incremental accuracy exceeds HyCal by 10.93 and 13.74 percentage points. A fixed-fusion centering ablation improves CIFAR L/14 novel accuracy by 41.73 points after independent offset recalibration; separately optimized normalization schemes confirm ImageNet-R capacity gains of 3.60 and 5.29 points. Under a common direct accuracy certificate, the original retained ImageNet-R L/14 readout reaches 81.61% novel accuracy, versus 76.76% for dense prototypes, with both admitted in all ten partitions. Frozen source models and prototypes reach 86.82% and 58.96% novel accuracy on ImageNet-Sketch after target calibration, with all evaluated empirical retention targets met. The contribution connects a concrete evidence-alignment mechanism to sustained recognition and useful admission. Code and experimental records accompany the manuscript.

The manuscript is original, has not been formally published and is not under consideration elsewhere. I approve submission to Knowledge-Based Systems, declare no competing interests, and received no external funding. AI assistance is disclosed. Code and source-hashed records are available at https://github.com/Cheerssom/Reference-Anchored-Candidate-Tables .

Sincerely,
Junyi Zhang
Shandong University, China
cheerssom0876@gmail.com
