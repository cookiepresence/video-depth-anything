mkdir -p /scratch/mde
mkdir -p /scratch/mde/vkitti2

./multipart_download https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar /scratch/mde/vkitti2/vkitti2_rgb.tar 8
./multipart_download https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_depth.tar /scratch/mde/vkitti2/vkitti2_depth.tar 8

tar -xvf /scratch/mde/vkitti2/vkitti2_rgb.tar -C /scratch/mde/vkitti2
tar -xvf /scratch/mde/vkitti2/vkitti2_depth.tar -C /scratch/mde/vkitti2
