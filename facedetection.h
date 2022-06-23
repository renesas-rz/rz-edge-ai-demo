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

#ifndef FACEDETECTION_H
#define FACEDETECTION_H

#include <QMainWindow>

#include <opencv2/videoio.hpp>

#include "edge-utils.h"

#define TEXT_FACE_MODEL "face_landmark.tflite"

class edgeUtils;

namespace Ui { class MainWindow; }

class faceDetection : public QObject
{
    Q_OBJECT

public:
    faceDetection(Ui::MainWindow *ui, QString inferenceEngine);
    void setCameraMode();
    void setImageMode();
    void setVideoMode();
    void setFrameDims(int height, int width);

public slots:
    void runInference(const QVector<float>& receivedTensor, int receivedStride, int receivedTimeElapsed, const cv::Mat &receivedMat);
    void stopContinuousMode();
    void triggerInference();

signals:
    void getFrame();
    void sendMatToView(const cv::Mat&receivedMat);
    void startVideo();
    void stopVideo();

private:
    void setButtonState(bool enable);
    QVector<float> sortTensorFaceLandmark(const QVector<float> receivedTensor, int receivedStride);
    void drawPointsFaceLandmark(const QVector<float>& outputTensor, bool updateGraphicalView);
    void connectLandmarks(int landmark1, int landmark2, bool drawGraphicalViewLandmarks);

    Ui::MainWindow *uiFD;
    Input inputModeFD;
    edgeUtils *utilFD;
    QVector<float> outputTensor;
    QVector<float> xCoordinate;
    QVector<float> yCoordinate;
    bool continuousMode;
    bool buttonState;
    int frameHeight;
    int frameWidth;
};

#endif // FACEDETECTION_H
