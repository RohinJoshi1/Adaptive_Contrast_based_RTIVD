To install dependencies:

```
 pip install -r requirements.txt
```

To run CPU:

```
  cd src/model
  python Dehazer.py
```

To run GPU:

```
cd src/model
python cuda_dehazer.py
```

To run test:

```
  cd benchmarks
  python tests.py
```

### Directory layout:

    benchmarks/tests.py
    input/...
    output/...
    src/
    model/
    gpu_model
    cpu_model

## Metrics:

## Our Method

### PSNR :

        1. 58.png = 27.53935125197399 dB
        2. 70.png = 27.68479667506341 dB
        3. 77.png = 27.77244962699312 dB

### SSIM :

        1. 58.png = 0.7108081616153676
        2. 70.png = 0.8455792651388446
        3. 77.png = 0.8738069935490138

## DADFNET

### PSNR :

      1. 58.png = 27.845728929426276 dB
      2. 70.png = 27.56345437073974 dB
      3. 77.png = 27.79549319105797 dB

### SSIM :

      1. 58.png = 0.8762965421548553
      2. 70.png = 0.8618195626894056
      3. 77.png = 0.8856375115279542
