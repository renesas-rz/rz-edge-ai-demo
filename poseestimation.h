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

#ifndef POSEESTIMATION_H
#define POSEESTIMATION_H

#include <QMainWindow>

#include <chrono>
#include <opencv2/videoio.hpp>

#include "edge-utils.h"

#define IDENTIFIER_MOVE_NET "lite-model_movenet_singlepose"
#define IDENTIFIER_BLAZE_POSE "pose_landmark"
#define IDENTIFIER_HAND_POSE "hand_landmark"

#define TEXT_LOAD_FILE "Load Image/Video"
#define TEXT_LOAD_NEW_FILE "Load New Image/Video"

namespace Ui { class MainWindow; }

enum PoseModel { MoveNet, BlazePose, HandPose };

class poseEstimation : public QObject
{
    Q_OBJECT

public:
    poseEstimation(Ui::MainWindow *ui, PoseModel poseModel);
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
    QVector<float> sortTensorMoveNet(const QVector<float> receivedTensor, int receivedStride);
    QVector<float> sortTensorBlazePose(const QVector<float> receivedTensor, int receivedStride);
    QVector<float> sortTensorHandPose(const QVector<float> receivedTensor, int receivedStride);
    void drawLimbsMoveNet(const QVector<float>& outputTensor, bool updateGraphicalView);
    void drawLimbsBlazePose(const QVector<float>& outputTensor, bool updateGraphicalView);
    void drawLimbsHandPose(const QVector<float>& outputTensor, bool updateGraphicalView);
    void connectLimbs(int limb1, int limb2, bool drawGraphicalViewLimbs);
    void displayTotalFps(int totalProcessTime);
    void timeTotalFps(bool startingTimer);

    Ui::MainWindow *uiPE;
    Input inputModePE;
    PoseModel poseModelSet;
    QVector<float> outputTensor;
    QVector<float> xCoordinate;
    QVector<float> yCoordinate;
    std::chrono::high_resolution_clock::time_point startTime;
    bool continuousMode;
    bool buttonState;
    int frameHeight;
    int frameWidth;
};

#endif // POSEESTIMATION_H
