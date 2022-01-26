# Development Notes

## Notes

## Commands

### Start GPUMon on GPU Instances

```bash
source activate pytorch_p38


# Essentially Exports nvidia-smi to Cloudwatch...

sudo pip install nvidia-ml-py boto3 pynvml &&\
    cd ~ &&\
    mkdir -p ./tools/GPUMonitoring/ &&
    curl -XGET https://s3.amazonaws.com/aws-bigdata-blog/artifacts/GPUMonitoring/gpumon.py > ./tools/GPUMonitoring/gpumon.py &&\
    python3 ./tools/GPUCloudWatchMonitor/gpumon.py
```

### Generate a Shuffled Sample

```bash
# Generic: shuf -zn500 -e ${SRC_LOCATION}/*.jpg | xargs -0 cp -vt ${TARGET_LOCATION}
# This is recommended to have a small sample of a few thousand images to run quick tests
# on. And then using `/efs/samples/001/` as DATA_ROOT when prompted...

sudo mkdir -p /efs/samples/001/

sudo shuf -zn500 -e /efs/images/*.jpg |\
    xargs -0 cp -vt /efs/samples/001/

# OR (Even lazier)

sudo cp /efs/images/a* /efs/samples/001/
```

### Sync From S3 -> EFS

```bash
aws s3 sync s3://dmw2151-path-to-imgs-bucket /efs/imgs

sudo aws s3 cp --recursive s3://dmw2151-street-images/imgs/ /efs/images/
```