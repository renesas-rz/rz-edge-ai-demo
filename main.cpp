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

#define OPTION_FD_DETECT_FACE "face"
#define OPTION_FD_DETECT_IRIS "iris"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QCommandLineParser parser;
    QCommandLineOption autoStartOption (QStringList() << "a" << "autostart", "Enable inference to automatically start when the application opens.");
    QCommandLineOption cameraOption(QStringList() << "c" << "camera", "Choose a camera.", "file");
    QCommandLineOption labelOption (QStringList() << "l" << "label", "Choose a label for selected demo mode.", "file");
    QCommandLineOption modelOption (QStringList() << "m" << "model", "Choose a model for selected demo mode.", "file");
    QCommandLineOption modeOption (QStringList() << "s" << "start-mode",
                                   "Choose a mode to start the application in: [shopping-basket|object-detection|pose-estimation|\nface-detection|audio-command].",
                                   "mode", QString("pose-estimation"));
    QCommandLineOption pricesOption (QStringList() << "p" << "prices-file",
                                   "Choose a text file listing the prices to use for the shopping basket mode", "file", PRICES_PATH_DEFAULT);
    QCommandLineOption faceDetectOption (QStringList() << "f" << "face-mode", "Choose a mode to start face detection with: [iris|face].", "mode");
    QCommandLineOption videoOption (QStringList() << "v" << "video-image", "Choose a video/image to load during startup. Displays before -c option during startup.", "media");
    bool autoStart;
    QString cameraLocation;
    QString labelLocation;
    QString modelLocation;
    QString modeString;
    QString pricesLocation;
    QString videoLocation;
    QString faceOption;
    QString boardName;
    bool irisOption = false;
    QSysInfo systemInfo;
    Mode mode = PE;
    QString applicationDescription =
    "Selecting Demo Mode\n"
    "Demo Mode->Object Detection: Object Detection Mode.\n"
    "Demo Mode->Shopping Basket: Shopping Basket Mode.\n"
    "Demo Mode->Pose Estimation: Pose Estimation Mode.\n"
    "Demo Mode->Face Detection: Face Detection Mode.\n"
    "Demo Mode->Audio Detection: Audio Command Mode.\n\n"
    "Required Hardware:\n"
    "  Camera: Currently the Google Coral Mipi camera is supported,\n"
    "          but should work with any UVC compatible USB camera.\n\n"
    "Object Detection Mode\n"
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
    "Pose Estimation Mode (Default)\n"
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
    "Audio Command Mode\n"
    "  Listens to audio based commands to move an arrow across a grid on screen.\n"
    "  This displays the inference time, command history and command count.\n"
    "  The go command moves the arrow one space in the direction it is pointing.\n"
    "  The off commands reset arrow position, command history and count.\n"
    "  The stop command halts recording from the audio source.\n\n"
    "Buttons:\n"
    " Talk: Run inference on the currently selected audio source.\n"
    " Input->Load Audio File: Load a .wav file.\n"
    "        .wav files must be:\n"
    "             44100hz\n"
    "             1 Second (44034 frames) duration\n"
    "             32 bit float data\n"
    "             RIFF (little-endian) format\n\n"
    "Common Mode Buttons:\n"
    "  Input->Use Camera: Removes media file and resumes live camera feed in supporting modes.\n"
    "  Inference Engine->TensorFlow Lite + ArmNN delegate: Run inference using TensorFlow\n"
    "                                                      Lite with ArmNN delegate enabled.\n"
    "  Inference Engine->TensorFlow Lite + XNNPack delegate: Run inference using TensorFlow\n"
    "                                                        Lite with XNNPack delegate enabled.\n"
    "  Inference Engine->TensorFlow Lite: Run inference using TensorFlow Lite.\n"
    "  About->Hardware: Display the platform information.\n"
    "  About->License: Read the license information.\n"
    "  About->Exit: Close the application.\n\n"
    "Default Options:\n"
    "  Camera: /dev/video0\n"
    "  Label:\n"
    "    Object Detection Mode: mobilenet_ssd_v2_coco_quant_postprocess_labels.txt\n"
    "    Shopping Basket Mode: shoppingBasketDemo_labels.txt\n"
    "    Audio Command Mode: audioDemo_labels.txt\n"
    "  Model:\n"
    "    Object Detection Mode: mobilenet_ssd_v2_coco_quant_postprocess.tflite\n"
    "    Shopping Basket Mode: shoppingBasketDemo.tflite\n"
    "    Pose Estimation Mode: pose_landmark_lite.tflite\n"
    "    Face Detection Mode: face_detection_short_range.tflite + face_landmark.tflite\n"
    "    Audio Command Mode: browserfft-speech-renesas.tflite\n"
    "  Mode Specific Files:\n"
    "    Shopping Basket Prices: shoppingBasketDemo_prices.txt\n\n"

    "Application Exit Codes:\n"
    "  0: Successful exit\n"
    "  1: Camera initialisation failed\n"
    "  2: Camera stopped working";
    QStringList supportedPoseModels = { MODEL_PATH_PE_MOVE_NET_L, MODEL_PATH_PE_MOVE_NET_T, MODEL_PATH_PE_BLAZE_POSE_FULL,
                                        MODEL_PATH_PE_BLAZE_POSE_HEAVY, MODEL_PATH_PE_BLAZE_POSE_LITE,
                                        MODEL_PATH_PE_HAND_POSE_FULL, MODEL_PATH_PE_HAND_POSE_LITE };

    parser.addOption(autoStartOption);
    parser.addOption(cameraOption);
    parser.addOption(labelOption);
    parser.addOption(modelOption);
    parser.addOption(modeOption);
    parser.addOption(pricesOption);
    parser.addOption(faceDetectOption);
    parser.addOption(videoOption);
    parser.addHelpOption();
    parser.setApplicationDescription(applicationDescription);
    parser.process(a);
    cameraLocation = parser.value(cameraOption);
    labelLocation = parser.value(labelOption);
    modelLocation = parser.value(modelOption);
    pricesLocation = parser.value(pricesOption);
    faceOption = parser.value(faceDetectOption);
    modeString = parser.value(modeOption);
    autoStart = parser.isSet(autoStartOption);
    videoLocation = parser.value(videoOption);

    boardName = systemInfo.machineHostName();

    /* Mode selection (-s / --start-mode) */
    if (modeString == "shopping-basket") {
        mode = SB;
    } else if (modeString == "object-detection") {
        mode = OD;
    } else if (modeString == "pose-estimation") {
        mode = PE;
    } else if (modeString == "face-detection") {
        mode = FD;

        if (!modelLocation.isEmpty())
            qWarning("Warning: face detection mode does not support loading of models, using default option...");

        if (faceOption == OPTION_FD_DETECT_FACE) {
            modelLocation = MODEL_PATH_FD_FACE_DETECTION;
            irisOption = false;
        } else if (faceOption == OPTION_FD_DETECT_IRIS) {
            modelLocation = MODEL_PATH_FD_IRIS_LANDMARK;
            irisOption = true;
        } else {
            if (!faceOption.isEmpty())
                qWarning("Warning: unknown face detection mode requested, using default option...");

            irisOption = false;
            modelLocation = MODEL_PATH_FD_FACE_DETECTION;
        }
    } else if (modeString == "audio-command") {
        mode = AC;
        modelLocation = MODEL_PATH_AC;
    } else {
        qWarning("Warning: unknown demo mode requested, starting in default mode...");
    }

    if (mode != FD) {
        if (!faceOption.isEmpty())
            qWarning("Warning: demo mode requested does not support face mode parameter...");

        faceOption = OPTION_FD_DETECT_FACE;
    }

    /* Label file selection (-l / --label)*/
    if (!QFileInfo(labelLocation).isFile()) {
        if (mode != PE && !labelLocation.isEmpty())
                qWarning("Warning: label file does not exist, using default label file...");

        if (mode == SB)
            labelLocation = LABEL_PATH_SB;
        else
            labelLocation = LABEL_PATH_OD;
    }

    /* Model file section (-m / --model) */
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

    /* Application start */
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    MainWindow w(nullptr, boardName, cameraLocation, labelLocation, modelLocation, videoLocation, mode, pricesLocation, irisOption, autoStart);
    w.show();
    return a.exec();
}
