# ResNet satellite image population estimator

## Procedure

### 1. Download population count dataset

- Source: GHSL (Global Human Settlement Layer)
- Product: GHS-POP
- Link: https://human-settlement.emergency.copernicus.eu/download.php?ds=pop
- Epoch: 2020
- Resolution: 3 arcsec
- Coordinate system: WGS84
- Extract zip to directory: `ghsl_data`

### 2. Prepare dataset

```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 generate_dataset.py
```

