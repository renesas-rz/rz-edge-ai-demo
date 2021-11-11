# RZ Edge AI Demo Application Source Code

This repository contains the code required to build the application demo. This demo requires [TensorFlow](https://github.com/tensorflow/tensorflow/tree/v2.3.1) and [OpenCV](https://opencv.org/).

## Manual Build Instructions
### RZ/G2
1. Setup the yocto enviroment described in [meta-rz-edge-ai-demo](https://github.com/renesas-rz/meta-rz-edge-ai-demo) (copy `.conf` files from templates) and run `bitbake core-image-qt-sdk -c populate_sdk` to create the cross toolchain.
2. Install cross toolchain with `sudo sh ./poky-glibc-x86_64-core-image-qt-sdk-aarch64-toolchain-<SDK Version>.sh`.
3. Set up environment variables with `source /<SDK location>/environment-setup-aarch64-poky-linux`.
4. Run `qmake`.
5. Run `make`.
6. Copy `rz-edge-ai-demo` to the root filesystem.
7. Copy `shoppingBasketDemo.tflite` to `/opt/rz-edge-ai-demo`.
8. Run the app with `./rz-edge-ai-demo`.

### Ubuntu
1. Install dependencies
    ```
    sudo apt install cmake qtbase5-dev qtdeclarative5-dev qt5-default qtmultimedia5-dev qtcreator
    ```

2. Install opencv core and opencv videoio, make sure your version has Gstreamer enabled. Otherwise build and install [OpenCV](https://github.com/opencv/opencv.git)
    ```
    git clone  https://github.com/opencv/opencv.git
    cd opencv/
    mkdir build/
    cd build/
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr \
    -D WITH_V4L=ON \
    -D WITH_QT=ON \
    -D WITH_GSTREAMER=ON ..
    make -j$(nproc)
    sudo make -j$(nproc) install
    ```

3. Build and install [TensorFlow lite v2.3.1](https://github.com/tensorflow/tensorflow/tree/v2.3.1)
    ```
    git clone  https://github.com/tensorflow/tensorflow.git
    cd tensorflow/
    git checkout v2.3.1
    ./tensorflow/lite/tools/make/download_dependencies.sh
    make -j$(nproc) -f ./tensorflow/lite/tools/make/Makefile
    sudo cp ./tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a /usr/local/lib/
    sudo cp -r tensorflow/ /usr/local/include
    sudo cp -r tensorflow/lite/tools/make/downloads/flatbuffers/include/flatbuffers /usr/local/include
    ```

4. Copy [shoppingBasketDemo.tflite](https://github.com/renesas-rz/meta-rz-edge-ai-demo/blob/master/recipes-ai/shopping-basket-mode/files/shoppingBasketDemo.tflite) to `/opt/rz-edge-ai-demo`
    ```
    sudo mkdir /opt/rz-edge-ai-demo
    cd /opt/rz-edge-ai-demo
    sudo wget https://github.com/renesas-rz/meta-rz-edge-ai-demo/raw/master/recipes-ai/shopping-basket-mode/files/shoppingBasketDemo.tflite
    ```

5. Exclude ArmNN incompatible code

   As ArmNN is not supported for x86 machines, remove the ArmNN code by uncommenting
   the line below from rz-edge-ai-demo.pro:
   ```
   #DEFINES += SBD_X86
   ```

6. Build demo application
    ```
    cd rz-edge-ai-demo
    qmake
    make -j$(nproc)
    sudo cp rz-edge-ai-demo /opt/rz-edge-ai-demo
    ```

7. Run the demo with `/opt/rz-edge-ai-demo/rz-edge-ai-demo`
