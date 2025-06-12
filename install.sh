#/bin/bash

sudo apt update && \
sudo apt upgrade && \

# vscode
echo "code code/add-microsoft-repo boolean true" | sudo debconf-set-selections &&\
sudo apt-get install wget gpg &&\
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg &&\
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg &&\
echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" |sudo tee /etc/apt/sources.list.d/vscode.list > /dev/null &&\
rm -f packages.microsoft.gpg &&\
sudo apt install apt-transport-https &&\
sudo apt update &&\
sudo apt install code  &&\   # or code-insiders 

# CUDA redirect

sudo ln -s /usr/local/cuda-11.4 /usr/local/cuda &&\
echo -e "\nexport PATH=/usr/lib/cuda/bin:\$PATH\nexport LD_LIBRARY_PATH=/usr/lib/cuda/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc &&\
source ~/.bashrc &&\

sudo apt install htop -y &&\  #htop install 

sudo apt install python3-pip libopenblas-base libopenmpi-dev &&\  # pip install

#PyTorch Install

wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl &&\  
sudo apt install libopenmpi-dev libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev  &&\  
pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl  &&\  
sudo ln -s /usr/lib/aarch64-linux-gnu/libmpi_cxx.so.40 /usr/lib/libmpi_cxx.so.20 &&\  
sudo ldconfig &&\  


#Scikit
pip install scikit-learn &&\  
pip install scikit-image  &&\  


echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc  &&\  
source ~/.bashrc  &&\  

#ROS
sudo apt update && sudo apt upgrade -y  &&\  
sudo apt install -y curl gnupg2 lsb-release  &&\  
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'  &&\  
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654  &&\  
sudo apt update  &&\  
sudo apt install -y ros-noetic-ros-base &&\  

sudo apt install python3-rosdep  &&\  
sudo rosdep init  &&\  
rosdep update  &&\  
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc  &&\  
source ~/.bashrc  &&\  
sudo apt install -y python3-rosinstall python3-rosinstall-generator python3-wstool build-essential  &&\

#tqdm and moviepy
pip3 install tqdm &&\
pip3 install "moviepy<2.0.0" &&\
