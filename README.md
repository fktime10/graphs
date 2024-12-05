## Train

```python
vqt2g_main.py --log-level INFO --comment "GVQVAE training with directed graph adjst" config/synthetic_config.yaml
```

## Inference

```python
recon_main.py runs/synthetic_test_2_Dec_04_20-15/config.yaml --num 30 --edge-sampling --threshold 0.5
```