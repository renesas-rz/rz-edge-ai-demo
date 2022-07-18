/*****************************************************************************************
 * Copyright (C) 2022 Renesas Electronics Corp.
 * This file is part of the RZ Edge AI Demo.
 *
 * The RZ Edge AI Demo is free software using the Qt Open Source Model: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * The RZ Edge AI Demo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with the RZ Edge AI Demo.  If not, see <https://www.gnu.org/licenses/>.
 *****************************************************************************************/

#include <QApplication>
#include <QCommandLineParser>
#include <QFileInfo>
#include <QSysInfo>

#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QCommandLineParser parser;
    QCommandLineOption cameraOption(QStringList() << "c" << "camera", "Choose a camera.", "file");
    QCommandLineOption labelOption (QStringList() << "l" << "label", "Choose a label for selected demo mode.", "file");
    QCommandLineOption modelOption (QStringList() << "m" << "model", "Choose a model for selected demo mode.", "file");
    QCommandLineOption modeOption (QStringList() << "s" << "start-mode",
                                   "Choose a mode to start the application in: [shopping-basket|object-detection|pose-estimation|\nface-detection].",
                                   "mode", QString("object-detection"));
    QCommandLineOption pricesOption (QStringList() << "p" << "prices-file",
                                   "Choose a text file listing the prices to use for the shopping basket mode", "file", PRICES_PATH_DEFAULT);
    QString cameraLocation;
    QString labelLocation;
    QString modelLocation;
    QString modeString;
    QString pricesLocation;
    QString boardName;
    QSysInfo systemInfo;
    Mode mode = OD;
    QString applicationDescription =
    "Selecting Demo Mode\n"
    "Demo Mode->Object Detection: Object Detection Mode.\n"
    "Demo Mode->Shopping Basket: Shopping Basket Mode.\n"
    "Demo Mode->Pose Estimation: Pose Estimation Mode.\n"
    "Demo Mode->Face Detection: Face Detection Mode.\n\n"
    "Required Hardware:\n"
    "  Camera: Currently the Google Coral Mipi camera is supported,\n"
    "          but should work with any UVC compatible USB camera.\n\n"
    "Object Detection Mode (Default)\n"
    "  Draws boxes around detected objects and displays the following:\n"
    "    - Object Name and confidence level\n"
    "    - Table of the detected objects\n"
    "    - Inference time\n"
    "    - Total FPS\n\n"
    "Buttons:\n"
    "  Load AI Model: Load a different model and label file.\n"
    "  Start Inference/Stop Inference: Starts the live camera feed/media file,\n"
    "                                  grabs the frame and runs inference or just\n"
    "                                  displays the live camera feed/media file.\n"
    "  Input->Load Image/Video: Load an image or video file.\n\n"
    "Shopping Basket Mode\n"
    "  Draws boxes around detected shopping items, displays the name\n"
    "  and confidence of the object, populates a checkout list, and\n"
    "  also displays inference time.\n\n"
    "Buttons:\n"
    "  Load AI Model: Load a different model, label file and prices file.\n"
    "  Process Basket: Pauses the live camera feed, grabs the frame and runs inference.\n"
    "  Next Basket: Clears inference results and resumes live camera feed.\n"
    "  Input->Load Image: Load a static image file.\n\n"
    "Pose Estimation Mode\n"
    "  BlazePose, MoveNet: Draws lines to connect identified joints and facial features,\n"
    "                      displays the total FPS, inference time, and displays a 2-D Point\n"
    "                      Projection of the identified pose.\n\n"
    "  HandPose: Draws lines to connect identified hand-knuckle points, displays the total FPS,\n"
    "            inference time, and displays a 2-D Point Projection of the identified features.\n\n"
    "Buttons:\n"
    "  Load AI Model: Load a different pose model. Currently supported: MoveNet, BlazePose,\n"
    "                                                                   HandPose.\n"
    "  Start Inference/Stop Inference: Starts the live camera feed/media file,\n"
    "                                  grabs the frame and runs inference or just\n"
    "                                  displays the live camera feed/media file.\n"
    "  Input->Load Image/Video: Load an image or video file.\n\n"
    "Face Detection Mode\n"
    "  Face Detection: Draws lines around the facial regions on the image frame, displays the\n"
    "                  total FPS and inference times for running face detection and face landmark\n"
    "                  models, and displays a 2-D Point Projection of the identified face mesh\n\n"
    "  Iris Detection: Draws a circle around the detected iris for both eyes on the image frame,\n"
    "                  displays the total FPS and inference times for running face detection, face\n"
    "                  landmark and iris landmark models, and displays a diagram showing the process\n"
    "                  from running face detection to the output of the iris landmark model.\n\n"
    "Buttons:\n"
    "  Detect Face: Detects the face of the identified person.\n"
    "  Detect Iris: Detects the irises of the identified person.\n"
    "  Start Inference/Stop Inference: Starts the live camera feed/media file,\n"
    "                                  grabs the frame and runs inference or just\n"
    "                                  displays the live camera feed/media file.\n"
    "  Input->Load Image/Video: Load an image or video file.\n\n"
    "Common Mode Buttons:\n"
    "  Input->Load Camera Feed: Removes media file and resumes live camera feed.\n"
    "  Inference Engine->TensorFlow Lite + ArmNN delegate: Run inference using TensorFlow\n"
    "                                                      Lite with ArmNN delegate enabled.\n"
    "  Inference Engine->TensorFlow Lite + XNNPack delegate: Run inference using TensorFlow\n"
    "                                                        Lite with XNNPack delegate enabled\n"
    "                                                        (RZ/G2L and RZ/G2LC only).\n"
    "  Inference Engine->TensorFlow Lite: Run inference using TensorFlow Lite.\n"
    "  About->Hardware: Display the platform information.\n"
    "  About->License: Read the license information.\n"
    "  About->Exit: Close the application.\n\n"
    "Default Options:\n"
    "  Camera: /dev/video0\n"
    "  Label:\n"
    "    Object Detection Mode: mobilenet_ssd_v2_coco_quant_postprocess_labels.txt\n"
    "    Shopping Basket Mode: shoppingBasketDemo_labels.txt\n"
    "  Model:\n"
    "    Object Detection Mode: mobilenet_ssd_v2_coco_quant_postprocess.tflite\n"
    "    Shopping Basket Mode: shoppingBasketDemo.tflite\n"
    "    Pose Estimation Mode: pose_landmark_lite.tflite\n"
    "    Face Detection Mode: face_detection_short_range.tflite + face_landmark.tflite\n"
    "  Mode Specific Files:\n"
    "    Shopping Basket Prices: shoppingBasketDemo_prices.txt\n\n"

    "Application Exit Codes:\n"
    "  0: Successful exit\n"
    "  1: Camera initialisation failed\n"
    "  2: Camera stopped working";
    QStringList supportedPoseModels = { MODEL_PATH_PE_MOVE_NET_L, MODEL_PATH_PE_MOVE_NET_T, MODEL_PATH_PE_BLAZE_POSE_FULL,
                                        MODEL_PATH_PE_BLAZE_POSE_HEAVY, MODEL_PATH_PE_BLAZE_POSE_LITE,
                                        MODEL_PATH_PE_HAND_POSE_FULL, MODEL_PATH_PE_HAND_POSE_LITE };

    parser.addOption(cameraOption);
    parser.addOption(labelOption);
    parser.addOption(modelOption);
    parser.addOption(modeOption);
    parser.addOption(pricesOption);
    parser.addHelpOption();
    parser.setApplicationDescription(applicationDescription);
    parser.process(a);
    cameraLocation = parser.value(cameraOption);
    labelLocation = parser.value(labelOption);
    modelLocation = parser.value(modelOption);
    pricesLocation = parser.value(pricesOption);
    modeString = parser.value(modeOption);

    boardName = systemInfo.machineHostName();

    if (modeString == "shopping-basket") {
        mode = SB;
    } else if (modeString == "object-detection") {
        mode = OD;
    } else if (modeString == "pose-estimation") {
        mode = PE;
    } else if (modeString == "face-detection") {
        mode = FD;

        if (!modelLocation.isEmpty())
            qWarning("Warning: face detection mode does not support loading of models, starting mode using Face Detection model...");

        modelLocation = MODEL_PATH_FD_FACE_DETECTION;
    } else {
        qWarning("Warning: unknown demo mode requested, starting in default mode...");
    }

    if (!QFileInfo(labelLocation).isFile()) {
        if (mode != PE && !labelLocation.isEmpty())
                qWarning("Warning: label file does not exist, using default label file...");

        if (mode == SB)
            labelLocation = LABEL_PATH_SB;
        else
            labelLocation = LABEL_PATH_OD;
    }

    if (!QFileInfo(modelLocation).isFile()) {
        if (mode != PE && !modelLocation.isEmpty())
            qWarning("Warning: AI model does not exist, using default AI model...");

        if (mode == SB)
            modelLocation = MODEL_PATH_SB;
        else if (mode == OD)
            modelLocation = MODEL_PATH_OD;
    }

    if (mode == PE && !(supportedPoseModels.contains(QFileInfo(modelLocation).absoluteFilePath()))) {
        if (!modelLocation.isEmpty())
            qWarning("Warning: unsupported pose model selected, using default Pose model");

        modelLocation = MODEL_PATH_PE_BLAZE_POSE_LITE;
    }

    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    MainWindow w(nullptr, boardName, cameraLocation, labelLocation, modelLocation, mode, pricesLocation);
    w.show();
    return a.exec();
}
