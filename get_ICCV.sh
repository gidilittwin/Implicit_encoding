
code_dir=`pwd`
mkdir /private/home/wolf/gidishape/data/ShapeNetICCV/
cd /private/home/wolf/gidishape/data/ShapeNetICCV/
wget 'https://shapenet.cs.stanford.edu/iccv17/recon3d/train_imgs.zip' -O train_imgs.zip
wget 'https://shapenet.cs.stanford.edu/iccv17/recon3d/train_voxels.zip' -O train_voxels.zip
wget 'https://shapenet.cs.stanford.edu/iccv17/recon3d/val_imgs.zip' -O val_imgs.zip
wget 'https://shapenet.cs.stanford.edu/iccv17/recon3d/val_voxels.zip' -O val_voxels.zip
wget 'https://shapenet.cs.stanford.edu/iccv17/recon3d/test_imgs.zip' -O test_imgs.zip
wget 'https://shapenet.cs.stanford.edu/iccv17/recon3d/test_voxels.zip' -O test_voxels.zip
unzip train_imgs.zip
unzip train_voxels.zip
unzip val_imgs.zip
unzip val_voxels.zip
unzip test_imgs.zip
unzip test_voxels.zip
mv test test_voxels
cd $code_dir
python proccess_data_iccv.py
