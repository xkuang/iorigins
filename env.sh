echo "export PATH=$PATH:/Users/ioanaveronicachelu/anaconda/bin" >> /Users/ioanaveronicachelu/.bash_profile
source /Users/ioanaveronicachelu/.bash_profile
conda create --name iorigins --clone root
source activate iorigins
echo "source activate iorigins" >> /Users/ioanaveronicachelu/.bash_profile

# update pip
pip install --upgrade pip
sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.7.1-cp27-none-any.whl
# pytube
pip install -r requirements.txt