# Dataset card — 6G AI channel estimation

- **File:** `data/datasets/channel_estimation_dataset.csv`
- **Size:** 80,000 rows (configurable in `config/system_config.json`)
- **Split:** 70% train / 15% validation / 15% test (`split` column, seed 42)

## Generative model

Channel amplitudes follow Rayleigh fading; LOS profiles (CDL-D/E, TDL-D/E) add a Ricean K-factor from TR 38.901. Delay spreads are drawn from scenario ranges inspired by TR 38.901 Table 7.5-6 and TS 38.101-4 TDL-A30/B100/C300. Doppler uses `v · fc / c`. NTN uses TR 38.811 delay-spread ranges; THz and RIS use research ranges beyond FR2-2.

LS is a noisy observation of the true coefficient. MMSE applies Wiener shrinkage with delay-spread correlation. The AI target residual is smaller, with Doppler and THz degrading the gain — matching the qualitative Rel-18 AI/ML CSI findings (TR 38.843).

Attacks inflate NMSE (worst for jamming and pilot contamination) and label the security tables.

Companion files: `mobility_dataset.csv`, `security_dataset.csv`, `digital_twin_states.csv`.
