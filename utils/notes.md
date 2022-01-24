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

sudo mkdir -p /efs/images/multi/b1

sudo shuf -zn500 -e /efs/images/*.jpg |\
    xargs -0 cp -vt /efs/images/b1

# OR:

ls -lh /efs/images | head -1000 | xargs -0 sudo cp -vt /efs/images/mutli/b1/

# OR (Even lazier)

sudo cp /efs/images/a* /efs/images/multi/b1/
```

### Sync From S3 -> EFS

```bash
aws s3 sync s3://dmw2151-path-to-imgs-bucket /efs/imgs

sudo aws s3 cp --recursive s3://dmw2151-street-images/imgs/ /efs/images/
```