# usage:
# ./download_pointodessey.sh <(s)ample/train-(p)arts/train-(f)ull/(t)est/(v)al>

mkdir -p /scratch/mde
mkdir -p /scratch/mde/pointodessey

ROOT=/scratch/mde/pointodessey

TAR_OPTIONS="--exclude */*/normals --exclude */*/masks --exclude */.mp4"

cd $ROOT

while getopts "spftv" OPT
do
      case $OPT in
	  # sample dataset
	  s) uvx gdown 1dnl9XMImdwKX2KcZCTuVDhcy5h8qzQIO --continue
	     tar -xvf sample.tar.gz $TAR_OPTIONS;;
	  # train dataset in 4 parts
	  p) uvx gdown "1iPXucKA3s5wLzYhm2VHSZbyW_-dFT1ar" --continue
	     uvx gdown 1nDesHqyKLV10dzQfZD9miQZWdtn3SjIn --continue
	     uvx gdown 1FU08bex4BKlYrpkGtK2FDpbwf1jI1bNg --continue
	     uvx gdown 1Q7Mw32PUMim6dTT6yNx0vatS6YOrow3Y --continue
	     cat train.tar.gz.part* > train.tar.gz
	     tar -xvf train.tar.gz $TAR_OPTIONS;;
	  f) uvx gdown 1ivaHRZV6iwxxH4qk8IAIyrOF9jrppDIP --continue
	     tar -xvf train.tar.gz $TAR_OPTIONS;;
	  t) uvx gdown "1jn8l28BBNw9f9wYFmd5WOCERH48-GsgB" --continue
	     tar -xvf test.tar.gz $TAR_OPTIONS;;
	  v) uvx gdown "10LRpiWbJItpGDgp9RnKLkvEbZShlK9D_" --continue
	     tar -xvf val.tar.gz $TAR_OPTIONS;;
	  ?) echo "Usage: $0 [spftv]"
      esac
done

# remove unnessary files
rm -rf $ROOT/*/*.mp4 

cd -
