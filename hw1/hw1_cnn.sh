wget https://www.dropbox.com/s/aclk15pt0u3esxy/cnn_rnn_model.zip?dl=0 -O cnn_model.zip
unzip cnn_model.zip
rm cnn_model.zip
python test_cnn.py $1 $2
