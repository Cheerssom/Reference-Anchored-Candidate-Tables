# Independent risk calibration

Run `scripts/run_kbs_independent_risk.ps1` from the repository root. Use `-Python` to select your environment. Existing complete partitions and sessions are skipped after source/input guards. Create `experiments/kbs_strengthening/PAUSE` to stop at the next saved unit; remove that file and rerun this script to resume. This stage reuses frozen feature caches and copies no datasets or model weights.

Verify with `python src/experiments/test_kbs_risk_control.py` and `python src/experiments/kbs_verify_independent_risk.py`. Rebuild tables with `python paper/kbs/build_independent_risk_results.py`.

Primary: retention r=0.95, trajectory confidence 0.95 for each prespecified method/order/partition under the stated independent class-conditional sampling assumptions. Finite-test attainment and useful certified expansion are reported separately. CIFAR/Pets return base-only fallback; ImageNet-R accepts finite thresholds for retained tables on every partition with both encoders. The protection criterion is sufficient and conservative for methods recovering reference errors. See `experiments/kbs_independent_risk_protocol.md` and `paper/kbs/independent_risk_report.md`.

Prediction archives preserve all methods, safety margins, confidence levels and fallback outcomes. Original data and source hashes are stored in each suite manifest. A real full rerun preserved all 1,320 per-partition/session checkpoint file hashes and modification times. New experiment records occupy about 184.5 MiB.
