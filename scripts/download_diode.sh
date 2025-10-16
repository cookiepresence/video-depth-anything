mkdir -p /scratch/mde

./multipart_download http://diode-dataset.s3.amazonaws.com/train.tar.gz /scratch/mde/diode_train.tar.gz 80
./multipart_download http://diode-dataset.s3.amazonaws.com/val.tar.gz   /scratch/mde/diode_val.tar.gz 80

tar -xvf /scratch/mde/diode_train.tar.gz
tar -xvf /scratch/mde/diode_val.tar.gz
