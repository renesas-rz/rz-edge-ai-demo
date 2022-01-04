#*****************************************************************************************
# Copyright (C) 2022 Renesas Electronics Corp.
# This file is part of the RZ Edge AI Demo.
#
# The RZ Edge AI Demo is free software using the Qt Open Source Model: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# The RZ Edge AI Demo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the RZ Edge AI Demo.  If not, see <https://www.gnu.org/licenses/>.
#*****************************************************************************************

QT += core gui multimedia widgets

CONFIG += c++14

# Uncomment the line below to build for the X86 architecture
#DEFINES += SBD_X86

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    opencvworker.cpp \
    shoppingbasket.cpp \
    tfliteworker.cpp \
    videoworker.cpp

HEADERS += \
    mainwindow.h \
    opencvworker.h \
    shoppingbasket.h \
    tfliteworker.h \
    videoworker.h

FORMS += \
    mainwindow.ui

INCLUDEPATH += \
    $$(SDKTARGETSYSROOT)/usr/include/opencv4 \
    $$(SDKTARGETSYSROOT)/usr/include/tensorflow/lite/tools/make/downloads/flatbuffers/include \
    $$(SDKTARGETSYSROOT)/usr/include/armnn \

LIBS += \
    -L $$(SDKTARGETSYSROOT)/usr/lib64 \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -lopencv_videoio \
    -ltensorflow-lite \
    -ldl \
    -lutil

!contains(DEFINES, SBD_X86) {
LIBS += \
    -larmnn \
    -larmnnDelegate \
    -larmnnUtils
}
