import sys
import glob
import h5py
import cv2

# TODO: Clean this; Obviously...

H5_FILE = "/data/imgs/h5/manilla_imgs.h5"
PATH = "/efs/imgs/train_val/manila/query/images/*.jpg"

n_files = len(glob.glob(PATH, recursive=True))

dt = h5py.special_dtype(vlen=np.dtype("uint8"))

# resize all images and load into a single dataset
with h5py.File(H5_FILE, "w") as h5f:

    msls = h5f.create_dataset(
        name="images",
        shape=(n_files,256, 256, 3),
        maxshape=(n_files, 256, 256, 3),
        dtype=dt,
    )

    for cnt, ifile in enumerate(glob.iglob(PATH)):
        img = cv2.imread(ifile, cv2.IMREAD_COLOR)
        img_resize = cv2.resize( img, (256, 256) )
        img_ds[cnt:cnt+1:,:,:] = img_resize
        