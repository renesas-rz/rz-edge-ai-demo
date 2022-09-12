# RZ Edge AI Demo Application Source Code

This repository contains the code required to build the RZ Edge AI Demo application.

## Manual Build Instructions
1. Setup the yocto environment described in [meta-rz-edge-ai-demo](https://github.com/renesas-rz/meta-rz-edge-ai-demo) for the target platform (copy `.conf` files from templates) and run `bitbake core-image-qt-sdk -c populate_sdk` to create the cross toolchain.
2. Install cross toolchain.

   RZ/G2E and RZ/G2M:
   ```
   sudo sh ./poky-glibc-x86_64-core-image-qt-sdk-aarch64-toolchain-<SDK Version>.sh
   ```

   RZ/G2L and RZ/G2LC:
   ```
   sudo sh ./poky-glibc-x86_64-core-image-qt-sdk-aarch64-<Platform>-toolchain-<SDK Version>.sh
   ```
   Replace `<Platform>` with `smarc-rzg2l` or `smarc-rzg2lc`.

3. Set up environment variables with `source /<SDK location>/environment-setup-aarch64-poky-linux`.
4. Run `qmake`.
5. Run `make`.
6. Copy `rz-edge-ai-demo` to the root filesystem.
7. Copy the directory `models/` to `/opt/rz-edge-ai-demo`.
8. Copy the directory `labels/` to `/opt/rz-edge-ai-demo`.
9. Run the app with `./rz-edge-ai-demo`.
