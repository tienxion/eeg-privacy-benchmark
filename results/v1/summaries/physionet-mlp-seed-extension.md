# PhysioNet MLP Seed-Extension Results

This report combines the PhysioNet subjects `1-46` MLP-posterior seed rows with the completed seed-extension rows. Lower membership-inference AUC is better for privacy; deltas are paired against same-seed plain EEGNet.

The aggregate values below are the frozen public record for this protocol.

## Completion

- `eegnet` extension seeds complete: `4`.
- `mixup_eegnet` extension seeds complete: `4`.
- `confidence_penalty_eegnet` extension seeds complete: `4`.
- `adversarial_eegnet` extension seeds complete: `4`.
- `csp_lda` extension seeds complete: `4`.

## Interpretation

- Mixup seeds evaluated: `8` (`13,17,19,23,29,31,37,41`).
- Mixup mean delta AUC versus EEGNet: `+0.0096`.
- Mixup privacy-improving same-seed comparisons: `2/8`.
- Mixup mean task balanced-accuracy delta: `+0.0014`.
- Conclusion: `not supported`.

## Aggregate Model Read

| Model | Seeds | Mean task BA | Mean AUC | Mean delta AUC | Privacy-improving seeds | Mean delta task BA | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `eegnet` | `13,17,19,23,29,31,37,41` | 0.6399 | 0.4951 | +0.0000 | 0/8 | +0.0000 | `baseline` |
| `mixup_eegnet` | `13,17,19,23,29,31,37,41` | 0.6413 | 0.5047 | +0.0096 | 2/8 | +0.0014 | `not supported` |
| `confidence_penalty_eegnet` | `13,17,19,23,29,31,37,41` | 0.6377 | 0.4961 | +0.0010 | 1/8 | -0.0023 | `not privacy-improving` |
| `adversarial_eegnet` | `13,17,19,23,29,31,37,41` | 0.5059 | 0.4989 | +0.0038 | 3/8 | -0.1340 | `not privacy-improving` |
| `csp_lda` | `13,17,19,23,29,31,37,41` | 0.6330 | 0.5059 | +0.0108 | 4/8 | -0.0069 | `not privacy-improving` |
