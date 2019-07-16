data_dir='../data'
data_id='12Z440hxJSGCIhCSYaX5tbvsQA61WD_RH'

mkdir $data_dir
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$data_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$data_id" -O $data_dir/X_lpd5.npy && rm -rf /tmp/cookies.txt

python multinn/prepare_data.py --data-file "$data_dir"/X_lpd5.npy --sample-size 40000