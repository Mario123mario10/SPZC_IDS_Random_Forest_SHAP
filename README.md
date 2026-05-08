# SPZC IDS Random Forest + SHAP

Projekt dotyczy implementacji i analizy systemu wykrywania intruzów na podstawie zbioru danych CICIDS2017. Celem jest przygotowanie danych, wytrenowanie modelu Random Forest oraz analiza istotności cech z wykorzystaniem metody SHAP.

## Wymagania

Projekt zakłada użycie następujących narzędzi:

- Python >= 3.10
- uv
- just

Narzędzia `uv` oraz `just` powinny być zainstalowane globalnie w systemie i dostępne z poziomu terminala.



## Struktura danych

Pełny zbiór CICIDS2017 nie jest przechowywany w repozytorium ze względu na rozmiar plików.

Należy pobrać pliki CSV ze strony Canadian Institute for Cybersecurity (https://cicresearch.ca/CICDataset/CIC-IDS-2017/) i umieścić je w katalogu:

```text
data_raw/

Po przetworzeniu dane zostaną zapisane w katalogu:

```text
data_processed/
```

Struktura katalogów:

```text
project/
├── data_raw/
├── data_processed/
├── scripts/
│   └── preprocess_data.py
├── pyproject.toml
├── uv.lock
├── justfile
└── README.md
```


## Instalacja zależności

Zalecany sposób instalacji zależności:

```bash
uv sync
```

Alternatywnie można użyć komendy z `justfile`:

```bash
just install
```

## Uruchomienie całego pipeline'u

```bash
just all
```