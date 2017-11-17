wget https://www.dropbox.com/s/75446q9kzjjtlj3/test_model.zip?dl=0 -O test_model.zip
unzip test_model.zip
rm test_model.zip
python test.py $1 all ./test_model/epoch_51 $2 $3
