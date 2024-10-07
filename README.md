To install dependencies:

```
 pip install -r requirements.txt
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

### GPU :

### PSNR:

1. 10.png = 28.25915651036749 dB
2. 14.png = 28.428803538053643 dB
3. 3.png = 28.217609982031927 dB
4. 48.png = 28.504184367680715 dB
5. 58.png = 27.84031576844953 dB
6. 70.png = 28.10374994999751 dB
7. 77.png = 27.98326625284464 dB

### SSIM :

1. 10.png = 0.8979719507206109
2. 14.png = 0.7497835130045117
3. 3.png = 0.9313763929658223
4. 48.png = 0.9310510527345303
5. 58.png = 0.7866550424070547
6. 70.png = 0.8876946959501971
7. 77.png = 0.8606463129393309

## DADFNET

### PSNR :

      1. 58.png = 27.845728929426276 dB
      2. 70.png = 27.56345437073974 dB
      3. 77.png = 27.79549319105797 dB

### SSIM :

      1. 58.png = 0.8762965421548553
      2. 70.png = 0.8618195626894056
      3. 77.png = 0.8856375115279542

## MAXIM

### PSNR :

      1. 58.png = 29.12483263657208 dB
      2. 70.png = 29.114470417745196 dB
      3. 77.png = 33.77208730171186 dB

### SSIM :

      1. 58.png = 0.9826838913602274
      2. 70.png = 0.8618195626894056
      3. 77.png = 0.9904056139179378
