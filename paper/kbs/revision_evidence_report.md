# Expanded class-partition and encoder evidence

Completed2026-10-22. The primary comparison now uses ten distinct class assignments on each of CIFAR-100, Oxford-IIIT Pets and ImageNet-R, with frozen OpenAI CLIP ViT-L/14 and ViT-B/32. Two orders and five arrivals are evaluated in each of60 partition runs; eight controlled readouts produce52,800 metric records. Ten class assignments share fixed images and are not ten independent datasets.

On ImageNet-R, retained-minus-dense-prototype average incremental gains are0.71 points [conditional paired95% bootstrap CI0.46,0.92] on L/14 and0.90 [0.69,1.11] on B/32. Retained-minus-class-mean-cache gains are1.42 [1.15,1.62] and1.71 [1.44,1.98]. CIFAR dense-prototype intervals span zero; Pets has no uniform retention advantage across the measured fixed-gate comparisons. All outcomes are retained.

Raw-score and fixed-temperature controls share the retained model, availability pattern and pseudo-episode grids. They also preserve old scores on append. Their comparison tests per-query branch normalization, not a claim that append preservation alone implies better recognition. The method is presented as a fusion/evaluation framework rather than a new end-to-end training algorithm. BiMC and HyCal are positioned from primary sources as related calibration work, without inventing executed baseline results.

The main manuscript leads with validation-selected admission and compares strong dense alternatives on the same assignments. ImageNet-R L/14 retained base/novel test accuracy is86.15%/84.63%, with10/10 validation-feasible partitions and2/10 meeting the separate95% test target. No finite-sample test-retention guarantee is inferred.

Independent archive reconstruction verifies all52,800 rows, shared targets and thresholds;5,940 old L/14 rows match exactly. Two simple-fusion numerical tests pass. B/32 uses existing cached weights; eight fresh re-embeddings per dataset have minimum cosine agreement above0.999998. Extraction starts at batch16 with OOM halving and uses the existing GPU mutex/two-thread configuration. No dependency rebuild or paid compute is used. Feature shards and session checkpoints support pause/resume.

Current PDFs:22-page main,29-page supplement; abstract183words. Main-plus-supplement review_complete.pdf contains all sections for review. The user-supplied reviewer score is informal and no journal revision is submitted. Local numerical/build checks are distinct from a fresh scientific referee review.

Reproduce: scripts/run_kbs_strengthening.ps1 -Stage Revision ; python src/experiments/kbs_verify_revision.py ; python paper/kbs/build_revision_results.py . Protocol: experiments/kbs_revision_protocol.md.
