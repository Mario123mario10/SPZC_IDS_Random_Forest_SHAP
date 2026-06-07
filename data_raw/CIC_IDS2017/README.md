# CIC_IDS2017 raw files

Place CICIDS2017 CSV files in this directory.

Required files for the final binary pipeline and legacy experiments:

```text
Monday-WorkingHours.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

How the final pipeline uses these files:

- Benign rows are used from all CICIDS2017 files.
- Attack rows are mapped to broad families such as `DDoS`, `DoS`,
  `BruteForce`, and `WebAttack`.
- `PortScan`, `Infiltration`, `Heartbleed`, `Bot`, and SQL injection labels are
  not part of the main binary model by default, although some legacy scripts
  may still use selected files for side experiments.

Run:

```bash
just preprocess-cicids2017
```
