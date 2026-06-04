# data_raw

Place the raw, unmodified CICIDS2017 and other dataset CSV files in this directory.

The full dataset files are not committed to the repository because they are too large.

Expected example files:

This is not an exhaustive list, but it includes files used by the main processing scripts. The scripts can be configured to use any file, but these are some of the files used in the predefined pipelines.

- `Monday-WorkingHours.pcap_ISCX.csv` (Used for Benign traffic)
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (Used for DDoS attacks in the `paper_baseline` variant)
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` (Used for PortScan attacks)
- `Tuesday-WorkingHours.pcap_ISCX.csv` (Contains FTP-Patator and SSH-Patator Brute Force attacks)
- `Brute Force -XSS.csv` (Can be used for Brute Force XSS attacks)
