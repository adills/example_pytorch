# setup for a PyTorch environment to use GPU

# make sure xcode is installed
xcode-select -v
echo "if xcode is not installed, run xcode-select --install"

brew install hdf5
brew install miniforge
pipenv --python 3.10
# install tensorflow-deps like 'conda install -c apple tensorflow-deps'
#pipenv install tensorflow-macos
#pipenv install tensorflow-metal
pipenv install torch torchvision torchaudio
pipenv run python gpu_test.py
