# CSE_CIC_IDS2018 raw files

Place CSE-CIC-IDS2018 CSV files in this directory.

Download source:

<https://data.mendeley.com/datasets/29hdbdzx2r/1>

Required files for external validation:

```text
Brute Force -Web.csv
Brute Force -XSS.csv
DDOS attack-HOIC.csv
DDOS attack-LOIC-UDP.csv
DDoS attacks-LOIC-HTTP.csv
DoS attacks-GoldenEye.csv
DoS attacks-Hulk.csv
DoS attacks-SlowHTTPTest.csv
DoS attacks-Slowloris.csv
FTP-BruteForce.csv
SSH-Bruteforce.csv
```

How the final pipeline uses these files:

- This dataset is never used for training.
- Rows are mapped to the same binary target: `Benign` = 0, `Attack` = 1.
- Original labels are preserved in metadata and grouped into attack families:
  `DDoS`, `DoS`, `BruteForce`, and `WebAttack`.
- Feature names are canonicalized to match the CICIDS2017-trained model, for
  example `Dst Port` and `Destination Port` are treated as the same feature.

Run this only after CICIDS2017 preprocessing and model training:

```bash
just preprocess-cse2018
```
