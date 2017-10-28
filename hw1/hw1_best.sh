wget https://www.dropbox.com/s/coqmfw10oh04b9k/best_model.zip?dl=0 -O best_model.zip
unzip best_model.zip
rm best_model.zip
python test_best.py $1 $2
