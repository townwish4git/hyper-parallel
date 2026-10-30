t129 / 8 x Ascend 910B3 / 100 optimizer steps
AdamW: corrected original TND attention; no BNSD workaround.
Muon: same configuration except optimizer; HP commit 867967f482bff8fe4f11279a9633053301d7900d; MF existing NS4 variant.
Images: upper row is raw total/LM/weighted-MTP loss; lower row is HP - MF at the same step.
No smoothing, step shift, or mean subtraction. MF accumulated loss is divided by two according to its native monitor.
Configuration paths are replaced with ${EXPERIMENT_ROOT} and ${MF_ROOT} for publication.
Both frameworks completed all100 steps on all8 ranks. These curves do not establish strict gradient-norm parity or long-run training quality.
