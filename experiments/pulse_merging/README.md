# P14 pulse aftereffect diagnostic

The job reads the three P14 training selections and their cleaned `SRTInIcePulses` series. It simulates merging on a fixed sample of 1,000 events per database. This equal-per-database sample describes the diagnostic population. It does not estimate a flux-weighted rate.

## Run on Phoenix

Upload the kit ZIP to the root of your PACE GraphNeT checkout. The kit includes the two diagnostic scripts and this guide. It uses the checkout's existing `comparison/contract.py` and `comparison/pilot_contract.json`.

```bash
cd /storage/project/r-itaboada3-0/rmaples3/software/graphnet
unzip -n pulse-diagnostic-kit.zip
sbatch experiments/pulse_merging/pace_diagnostic.sbatch
```

On a repeat deployment, compare the existing scripts with the kit before replacing them. The launcher uses the same data, selections, Python environment and account as the epoch-31 job. It requests the Phoenix `cpu-small` partition and one CPU for up to two hours. You can supply account or partition overrides to `sbatch` if required by your allocation. For other data or selections, export `SOURCE_DATA_ROOT` or `PILOT_SELECTION_ROOT` before submission. For another shared results location, export `PULSE_RESULTS_ROOT`.

Choose alternative windows with, for example, `sbatch experiments/pulse_merging/pace_diagnostic.sbatch --windows-ns 5 15 30`. Use `--aftereffect-column NAME` only if you have independently confirmed that an existing pulse-level column is 1 for the target aftereffect and 0 for other pulses. Event-level truth or noise flags alone do not identify pulse origins.

## Bring the results back

After the job finishes, download `comparison_runs/pulse_diagnostic/pulse-diagnostic-JOBID.zip` from the checkout (or your configured results location). Attach that ZIP to the chat. The log `pulse-diagnostic-JOBID.out` prints the absolute result path and SHA256 on success. If the job fails, bring back the log.

## Interpret the measurements

* **Adjacent pairs** show the time between consecutive pulses on one sensor and the charge of each pulse. **Brightest neighbors** show the signed time and charge of every other pulse relative to the brightest on that sensor in the same event. The earliest pulse wins a brightest-charge tie. Compare before and after in common brightness ranges. The observed distribution depends on the acquisition window, genuine later photons and cleaning. Its time bins have unequal widths.
* **Brightest timing profile** counts smaller companion pulses before and after the brightest pulse in matched delay bins, per 100 sensor/event anchors, including anchors without companions. It separates brightest charge into exploratory groups below 1 PE, 1–<5 PE and ≥5 PE. Zero-delay ties are available in the raw neighbor table. The rate denominator counts anchors and does not correct for observation time. An excess is a lead for checking waveforms or simulation truth.
* **Window summary and event windows** simulate inclusive 5, 10, 25, 50 and 100 ns windows anchored at the first pulse in each group. Zero is an identity control even for equal times. Each group keeps the first time and total charge. The tables report removed pulse rows, charge assigned to the surviving row, sensors hit, and changes relative to the 256-pulse cap. No input database or reconstruction is altered.
* **Timing diagnostic** plots the two timing views and merge fraction. The delayed-pulse view extends past 11 microseconds independently of the short merge windows. PMT afterpulses and short waveform-decomposition artifacts can have different timing signatures. The [IceCube PMT characterization](https://arxiv.org/abs/1002.2442) describes laboratory measurements.
* **Manifest** records sampled database hashes, schema, sensor identity fields, training-selection hashes, code hashes, units, settings, software and result-member hashes. If hardware keys are absent, exact DOM coordinates serve as a position proxy for the single-PMT P14 geometry. Check the recorded key before interpreting the result.
* Without verified pulse-origin labels, the job measures candidate populations and the effects of merging. With verified labels, it also counts labeled rows absorbed, other rows absorbed, precision and recall. These describe row consolidation. Charge from an absorbed aftereffect is still present in the surviving row. Identifying and rejecting false charge needs a separate labeled or waveform-level check.
