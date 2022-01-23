# Development Notes

## Notes

## Commands

### Start GPUMon on GPU Instances

```bash
source activate pytorch_p38

sudo pip install nvidia-ml-py -y boto3 &&\
    python3 /home/ubuntu/tools/GPUCloudWatchMonitor/gpumon.py
```

### Generate a Shuffled Sample

```bash
# Generic: shuf -zn500 -e ${SRC_LOCATION}/*.jpg | xargs -0 cp -vt ${TARGET_LOCATION}

shuf -zn500 -e /efs/images/train_val/london/query/images/*.jpg |\
    xargs -0 cp -vt ./efs/images/mutli_city/london
```
