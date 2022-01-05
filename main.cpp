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
#include <QFile>

#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QCommandLineParser parser;
    QCommandLineOption cameraOption(QStringList() << "c" << "camera", "Choose a camera.", "file");
    QCommandLineOption labelOption (QStringList() << "l" << "label", "Choose a label for Object Detection Mode.", "file");
    QCommandLineOption modelOption (QStringList() << "m" << "model", "Choose a model for Object Detection Mode.", "file");
    QString cameraLocation;
    QString labelLocation;
    QString modelLocation;
    QString applicationDescription =
    "Selecting Demo Mode\n"
    "Mode->Object Detection: Object Detection Mode.\n"
    "Mode->Shopping Basket: Shopping Basket Mode.\n\n"
    "Required Hardware:\n"
    "  Camera: Currently the Google Coral Mipi camera is supported,\n"
    "          but should work with any UVC compatible USB camera.\n\n"
    "Object Detection Mode (Default)\n"
    "  Draws boxes around detected objects, displays the name\n"
    "  and confidence level in the object, and also displays inference time\n"
    "  and Total FPS.\n\n"
    "Buttons:\n"
    "  Start/Stop: Starts the live camera feed, grabs the frame and runs inference\n"
    "              or just displays the live camera feed.\n\n"
    "Shopping Basket Mode\n"
    "  Draws boxes around detected shopping items, displays the name\n"
    "  and confidence of the object, populates a checkout list, and\n"
    "  also displays inference time.\n\n"
    "Buttons:\n"
    "  Process Basket: Pauses the live camera feed, grabs the frame and runs inference.\n"
    "  Next Basket: Clears inference results and resumes live camera feed.\n\n"
    "Common Mode Buttons:\n"
    "  About->Hardware: Display the platform information.\n"
    "  About->License: Read the license information.\n"
    "  About->Exit: Close the application.\n"
    "  Inference->Enable/Disable: Enable or disable the ArmNN Delegate\n"
    "                             during inference.\n\n"
    "Default Options:\n"
    "  Camera: /dev/video0\n\n"
    "Application Exit Codes:\n"
    "  0: Successful exit\n"
    "  1: Camera initialisation failed\n"
    "  2: Camera stopped working";

    parser.addOption(cameraOption);
    parser.addOption(labelOption);
    parser.addOption(modelOption);
    parser.addHelpOption();
    parser.setApplicationDescription(applicationDescription);
    parser.process(a);
    cameraLocation = parser.value(cameraOption);
    labelLocation = parser.value(labelOption);
    modelLocation = parser.value(modelOption);

    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    MainWindow w(nullptr, cameraLocation, labelLocation, modelLocation);
    w.show();
    return a.exec();
}
