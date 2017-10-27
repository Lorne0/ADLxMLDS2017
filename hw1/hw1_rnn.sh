wget https://www.dropbox.com/s/cx1nrr5z870sh3u/rnn_model.zip?dl=0 -O rnn_model.zip
unzip rnn_model.zip
rm rnn_model.zip
python test_rnn.py $1 $2
