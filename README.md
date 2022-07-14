# SAV_conv
 Spatial-Angular Versatile Convolution for Light Field Reconstruction.

 ## Angular SR

 There are four angular SR modes: inter28, inter27, extra1, and extra2.

 ### Test

 1. Change the current directory to *./LFASR_SAV*.
 2. Download the test datasets from XXX and put all the files into the directory *./TestData/*.
 3. Download the checkpoints from XXX and put all the files into the directory *./checkpoint/*.
 4. Run the script *./LFASR_testing.py*. An example:
   ```shell
   python LFASR_testing.py --mode="inter28" --save-flag
   ```
## Spatial SR

There are two benchmark datasets: [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR) and [CityU](https://github.com/jingjin25/LFSSR-ATO).

### Test

1. Change the current directory to *./LFSSR_SAV/*.
2. Download the test datasets from XXX and put all the files into the directory *./TestData/*.
3. Download the checkpoints from XXX and put all the files into the directory *./checkpoints/*.
4. If you want to test on the benchmark BasicLFSR, please run the script *./LFSSR_testing_BasicLFSR.py*. An example:
```shell
python LFSSR_testing_BasicLFSR.py --scale=4 --save-flag
```
5. If you want to test on the benchmark CityU, please run the script *./LFSSR_testing_CityU.py*. An example:
```shell
python LFSSR_testing_CityU.py --scale=4 --save-flag
```
