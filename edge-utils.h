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

#ifndef EDGEUTILS_H
#define EDGEUTILS_H

#include <QObject>

#include <chrono>

#define MODEL_PATH_FD_FACE_DETECTION "/opt/rz-edge-ai-demo/models/face_detection_short_range.tflite"

#define TEXT_INFERENCE "Inference Time: "
#define TEXT_LOAD_FILE "Load Image/Video"
#define TEXT_LOAD_NEW_FILE "Load New Image/Video"
#define TEXT_TOTAL_FPS "Total FPS: "

#define DETECT_DEFAULT_THRESHOLD 0.5

#define THEME_GREEN Qt::green
#define THEME_RED Qt::red
#define THEME_BLUE Qt::blue
#define THEME_GREEN Qt::green
#define BUTTON_BLUE "background-color: rgba(42, 40, 157);color: rgb(255, 255, 255);border: 2px;border-radius: 55px;border-style: outset;"
#define BUTTON_GREYED_OUT "background-color: rgba(42, 40, 157, 90);color: rgb(255, 255, 255);border: 2px;border-radius: 55px;border-style: outset;"
#define BUTTON_RED "background-color: rgba(255, 0, 0);color: rgb(255, 255, 255);border: 2px;border-radius: 55px;border-style: outset;"
#define BOX_POINTS 4

#define PEN_THICKNESS 2

#define GRAPHICS_VIEW_HEIGHT 600
#define GRAPHICS_VIEW_WIDTH 800

#define STACK_WIDGET_INDEX_SB 0
#define STACK_WIDGET_INDEX_OD 1
#define STACK_WIDGET_INDEX_PE 2
#define STACK_WIDGET_INDEX_FD 3
#define STACK_WIDGET_INDEX_AC 4

enum Board { G2E, G2L, G2LC, G2M, Unknown };
enum Input { cameraMode, imageMode, videoMode, audioFileMode };
enum Mode { SB, OD, PE, FD, AC };
enum FaceModel { faceDetect, faceLandmark, irisLandmarkL, irisLandmarkR };

class edgeUtils : public QObject
{
    Q_OBJECT

public:
    edgeUtils();
    void timeTotalFps(bool startingTimer);
    float calculateTotalFps();
    static float calculateSigmoid(float realNumber);

    std::chrono::high_resolution_clock::time_point startTime;
    int totalProcessTime;
};

#endif // EDGEUTILS_H
