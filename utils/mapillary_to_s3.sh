#! /bin/bash

# Script to Download Files from Mapillary Street Level Sequences -> EFS (local) and S3
# Sample usage:
#
# bash -x ./mapillary_to_s3.sh ${A_SIGNED_DOWNLOAD_URL} dmw2151-street-images

# NOTE: Download && Unzip Mapillary Datasets; send to S3 and EFS (assumes /efs/ mounted)
dl="$(uuidgen | tr -d '-').zip"
s3_bucket=$2

curl -XGET $1 --output $dl &&\
    sudo mkdir -p /efs/images &&\
    sudo unzip $dl -d /efs/images

# Send Compressed File to S3 (will have a randomized uuid) && Sync imgs directory to S3
aws s3 cp $dl s3://$s3_bucket/raw/ &&\
    rm $dl

s3 sync ./efs/images s3://$s3_bucket/imgs/

exit 0