#! /bin/bash

# Script to Download Files from Mapillary Street Level Sequences -> EFS (local) and S3
# Sample usage:
#
# bash -x ./mapillary_to_s3.sh ${A_SIGNED_DOWNLOAD_URL} dmw2151-street-images

# NOTE: Download && Unzip Mapillary Datasets; send to S3 and EFS (assumes /efs/ mounted)
dl="$(uuidgen | tr -d '-').zip"
s3_bucket=$2

curl -k -XGET $1 --output $dl &&\
    sudo mkdir -p /efs/images &&\
    sudo unzip $dl -d /efs/images

# Send Compressed File to S3 so we don't have to hammer MSLS bandwith
aws s3 cp $dl s3://$s3_bucket/raw/ && rm $dl

exit 0