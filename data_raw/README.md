# data_raw

This directory stores raw CSV datasets used by the project. Raw CSV files are
not committed to git because they are large.

Expected layout:

```text
data_raw/
  CIC_IDS2017/
    README.md
    *.pcap_ISCX.csv
  CSE_CIC_IDS2018/
    README.md
    *.csv
```

The final binary IDS pipeline uses:

- `data_raw/CIC_IDS2017/` for training and internal testing.
- `data_raw/CSE_CIC_IDS2018/` only for external validation.

See the README file inside each subdirectory for the exact expected filenames.
