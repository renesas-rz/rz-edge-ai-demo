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

#include "poseestimation.h"
#include "ui_mainwindow.h"

#include <cmath>
#include <math.h>

#include <QEventLoop>
#include <QGraphicsScene>
#include <QGraphicsTextItem>

#define BLAZE_POSE_INPUT_SIZE 256.0

#define DETECT_THRESHOLD 0.3

enum MoveNetPoints { NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER,
                     LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_KNEE,
                     RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE};

enum BlazePosePoints { BP_NOSE, BP_LEFT_EYE_INNER, BP_LEFT_EYE, BP_LEFT_EYE_OUTER, BP_RIGHT_EYE_INNER,
                       BP_RIGHT_EYE, BP_RIGHT_EYE_OUTER, BP_LEFT_EAR, BP_RIGHT_EAR, BP_LEFT_MOUTH,
                       BP_RIGHT_MOUTH, BP_LEFT_SHOULDER, BP_RIGHT_SHOULDER, BP_LEFT_ELBOW, BP_RIGHT_ELBOW,
                       BP_LEFT_WRIST, BP_RIGHT_WRIST, BP_LEFT_PINKY, BP_RIGHT_PINKY, BP_LEFT_INDEX, BP_RIGHT_INDEX,
                       BP_LEFT_THUMB, BP_RIGHT_THUMB, BP_LEFT_HIP, BP_RIGHT_HIP, BP_LEFT_KNEE, BP_RIGHT_KNEE,
                       BP_LEFT_ANKLE, BP_RIGHT_ANKLE, BP_LEFT_HEEL, BP_RIGHT_HEEL, BP_LEFT_FOOT_INDEX,
                       BP_RIGHT_FOOT_INDEX };

#define PEN_WIDTH 2
#define DOT_COLOUR Qt::green
#define LINE_COLOUR Qt::red

poseEstimation::poseEstimation(Ui::MainWindow *ui, PoseModel poseModel)
{
    uiPE = ui;
    inputModePE = cameraMode;
    poseModelSet = poseModel;
    buttonState = true;

    uiPE->actionShopping_Basket->setDisabled(false);
    uiPE->actionObject_Detection->setDisabled(false);
    uiPE->actionPose_Estimation->setDisabled(true);
    uiPE->actionLoad_Camera->setDisabled(true);
    uiPE->actionLoad_File->setText(TEXT_LOAD_FILE);

    uiPE->labelInference->setText(TEXT_INFERENCE);
    uiPE->labelDemoMode->setText("Mode: Pose Estimation");
    uiPE->labelTotalFpsPose->setText(TEXT_TOTAL_FPS);

    uiPE->stackedWidgetLeft->setCurrentIndex(STACK_WIDGET_INDEX_PE);
    uiPE->stackedWidgetRight->setCurrentIndex(STACK_WIDGET_INDEX_PE);

    QGraphicsScene *scenePointProjection = new QGraphicsScene(this);
    uiPE->graphicsViewPointProjection->setScene(scenePointProjection);
}

void poseEstimation::setButtonState(bool enable)
{
    if (enable) {
        buttonState = true;
        uiPE->pushButtonStartStopPose->setText("Start\nInference");
        uiPE->pushButtonStartStopPose->setStyleSheet(BUTTON_BLUE);
    } else {
        buttonState = false;
        uiPE->pushButtonStartStopPose->setText("Stop\nInference");
        uiPE->pushButtonStartStopPose->setStyleSheet(BUTTON_RED);
    }

    qApp->processEvents(QEventLoop::WaitForMoreEvents);
}

QVector<float> poseEstimation::sortTensorMoveNet(const QVector<float> receivedTensor, int receivedStride)
{
    QVector<float> sortedTensor = QVector<float>();

    float nanValue = std::nanf("NAN");

    for(int i = 0; i < receivedStride; i += 3) {
        float confidenceLevel = receivedTensor.at(i + 2);

        if (confidenceLevel > DETECT_THRESHOLD && confidenceLevel <= 1.0) {
            sortedTensor.push_back(receivedTensor.at(i));     // y-coordinate
            sortedTensor.push_back(receivedTensor.at(i + 1)); // x-coordinate
            sortedTensor.push_back(receivedTensor.at(i + 2)); // confidence
        } else {
            sortedTensor.push_back(nanValue);
            sortedTensor.push_back(nanValue);
            sortedTensor.push_back(nanValue);
        }
     }

    return sortedTensor;
}
QVector<float> poseEstimation::sortTensorBlazePose(const QVector<float> receivedTensor, int receivedStride)
{
    QVector<float> sortedTensor = QVector<float>();

    float nanValue = std::nanf("NAN");

    for(int i = 0; i < receivedStride; i += 5) {
        float confidenceValue = receivedTensor.at(i + 4);

        /* Calculate confidence probability by applying sigmoid function */
        float confidenceLevel = 1 / (1 + exp(-confidenceValue));

        if (confidenceLevel > 0.5 && confidenceLevel <= 1.0) {
            sortedTensor.push_back(receivedTensor.at(i + 1)); // y-coordinate
            sortedTensor.push_back(receivedTensor.at(i));     // x-coordinate
            sortedTensor.push_back(confidenceLevel);          // presence confidence
        } else {
            sortedTensor.push_back(nanValue);
            sortedTensor.push_back(nanValue);
            sortedTensor.push_back(nanValue);
        }
     }

    return sortedTensor;
}

void poseEstimation::drawLimbsMoveNet(const QVector<float> &outputTensor, bool updateGraphicalView)
{
    QPen pen;
    xCoordinate = QVector<float>();
    yCoordinate = QVector<float>();
    int displayWidth;
    int displayHeight;

    QGraphicsScene *scene = uiPE->graphicsView->scene();
    QGraphicsScene *scenePointProjection = uiPE->graphicsViewPointProjection->scene();

    pen.setWidth(PEN_WIDTH);

    if (updateGraphicalView) {
        /* Scale the dimensions down by 2 */
        displayHeight = frameHeight / 2;
        displayWidth = frameWidth / 2;

        scenePointProjection->clear();
    } else {
        displayHeight = frameHeight;
        displayWidth = frameWidth;
    }

    /* Save x and y coordinates in separate vectors */
    for (int i = 0; i < outputTensor.size(); i += 3) {
        float y = outputTensor[i] * float(displayHeight);
        float x = outputTensor[i + 1] * float(displayWidth);

        xCoordinate.push_back(x);
        yCoordinate.push_back(y);
    }

    /* Draw lines between the joints */
    connectLimbs(NOSE, LEFT_EYE, updateGraphicalView);
    connectLimbs(NOSE, RIGHT_EYE, updateGraphicalView);
    connectLimbs(LEFT_EYE, LEFT_EAR, updateGraphicalView);
    connectLimbs(RIGHT_EYE, RIGHT_EAR, updateGraphicalView);
    connectLimbs(LEFT_SHOULDER, RIGHT_SHOULDER, updateGraphicalView);
    connectLimbs(LEFT_SHOULDER, LEFT_ELBOW, updateGraphicalView);
    connectLimbs(LEFT_SHOULDER, LEFT_HIP, updateGraphicalView);
    connectLimbs(RIGHT_SHOULDER, RIGHT_ELBOW, updateGraphicalView);
    connectLimbs(RIGHT_SHOULDER, RIGHT_HIP, updateGraphicalView);
    connectLimbs(LEFT_ELBOW, LEFT_WRIST, updateGraphicalView);
    connectLimbs(RIGHT_ELBOW, RIGHT_WRIST, updateGraphicalView);
    connectLimbs(LEFT_HIP, RIGHT_HIP, updateGraphicalView);
    connectLimbs(LEFT_HIP, LEFT_KNEE, updateGraphicalView);
    connectLimbs(LEFT_KNEE, LEFT_ANKLE, updateGraphicalView);
    connectLimbs(RIGHT_HIP, RIGHT_KNEE, updateGraphicalView);
    connectLimbs(RIGHT_KNEE, RIGHT_ANKLE, updateGraphicalView);

    /* Draw dots on each detected joint */
    for (int i = 0; i <= RIGHT_ANKLE; i ++) {
        QBrush brush;

        float x = xCoordinate[i];
        float y = yCoordinate[i];

        pen.setColor(DOT_COLOUR);

        if (x >= 0 && y >= 0) {
            if (updateGraphicalView)
                scenePointProjection->addEllipse(x, y, PEN_WIDTH, PEN_WIDTH, pen, brush);
            else
                scene->addEllipse(x, y, PEN_WIDTH, PEN_WIDTH, pen, brush);
        }
    }
}

void poseEstimation::drawLimbsBlazePose(const QVector<float> &outputTensor, bool updateGraphicalView)
{
    QPen pen;
    xCoordinate = QVector<float>();
    yCoordinate = QVector<float>();
    int displayWidth;
    int displayHeight;

    QGraphicsScene *scene = uiPE->graphicsView->scene();
    QGraphicsScene *scenePointProjection = uiPE->graphicsViewPointProjection->scene();

    pen.setWidth(PEN_WIDTH);

    if (updateGraphicalView) {
        /* Scale the dimensions down by 2 */
        displayHeight = frameHeight / 2;
        displayWidth = frameWidth / 2;

        scenePointProjection->clear();
    } else {
        displayHeight = frameHeight;
        displayWidth = frameWidth;
    }

    float widthMultiplier = float(displayWidth) / BLAZE_POSE_INPUT_SIZE;
    float heightMultiplier = float(displayHeight) / BLAZE_POSE_INPUT_SIZE;

    /* Save x and y coordinates in separate vectors */
    for (int i = 0; i < outputTensor.size(); i += 3) {
        float y = outputTensor[i] * heightMultiplier;
        float x = outputTensor[i + 1] * widthMultiplier;

        xCoordinate.push_back(x);
        yCoordinate.push_back(y);
    }

    /* Draw lines between the joints */
    connectLimbs(BP_NOSE, BP_LEFT_EYE_INNER, updateGraphicalView);
    connectLimbs(BP_LEFT_EYE_INNER, BP_LEFT_EYE, updateGraphicalView);
    connectLimbs(BP_LEFT_EYE, BP_LEFT_EYE_OUTER, updateGraphicalView);
    connectLimbs(BP_LEFT_EYE_OUTER, BP_LEFT_EAR, updateGraphicalView);
    connectLimbs(BP_NOSE, BP_RIGHT_EYE_INNER, updateGraphicalView);
    connectLimbs(BP_RIGHT_EYE_INNER, BP_RIGHT_EYE, updateGraphicalView);
    connectLimbs(BP_RIGHT_EYE, BP_RIGHT_EYE_OUTER, updateGraphicalView);
    connectLimbs(BP_RIGHT_EYE_OUTER, BP_RIGHT_EAR, updateGraphicalView);
    connectLimbs(BP_NOSE, BP_LEFT_EYE, updateGraphicalView);
    connectLimbs(BP_LEFT_MOUTH, BP_RIGHT_MOUTH, updateGraphicalView);
    connectLimbs(BP_LEFT_SHOULDER, BP_RIGHT_SHOULDER, updateGraphicalView);
    connectLimbs(BP_LEFT_SHOULDER, BP_LEFT_ELBOW, updateGraphicalView);
    connectLimbs(BP_LEFT_ELBOW, BP_LEFT_WRIST, updateGraphicalView);
    connectLimbs(BP_LEFT_WRIST, BP_LEFT_THUMB, updateGraphicalView);
    connectLimbs(BP_LEFT_WRIST, BP_LEFT_INDEX, updateGraphicalView);
    connectLimbs(BP_LEFT_WRIST, BP_LEFT_PINKY, updateGraphicalView);
    connectLimbs(BP_LEFT_INDEX, BP_LEFT_PINKY, updateGraphicalView);
    connectLimbs(BP_RIGHT_SHOULDER, BP_RIGHT_ELBOW, updateGraphicalView);
    connectLimbs(BP_RIGHT_ELBOW, BP_RIGHT_WRIST, updateGraphicalView);
    connectLimbs(BP_RIGHT_WRIST, BP_RIGHT_THUMB, updateGraphicalView);
    connectLimbs(BP_RIGHT_WRIST, BP_RIGHT_INDEX, updateGraphicalView);
    connectLimbs(BP_RIGHT_WRIST, BP_RIGHT_PINKY, updateGraphicalView);
    connectLimbs(BP_RIGHT_INDEX, BP_RIGHT_PINKY, updateGraphicalView);
    connectLimbs(BP_LEFT_SHOULDER, BP_LEFT_HIP, updateGraphicalView);
    connectLimbs(BP_RIGHT_SHOULDER, BP_RIGHT_HIP, updateGraphicalView);
    connectLimbs(BP_LEFT_HIP, BP_RIGHT_HIP, updateGraphicalView);
    connectLimbs(BP_LEFT_HIP, BP_LEFT_KNEE, updateGraphicalView);
    connectLimbs(BP_LEFT_KNEE, BP_LEFT_ANKLE, updateGraphicalView);
    connectLimbs(BP_LEFT_ANKLE, BP_LEFT_HEEL, updateGraphicalView);
    connectLimbs(BP_LEFT_ANKLE, BP_LEFT_FOOT_INDEX, updateGraphicalView);
    connectLimbs(BP_LEFT_HEEL, BP_LEFT_FOOT_INDEX, updateGraphicalView);
    connectLimbs(BP_RIGHT_HIP, BP_RIGHT_KNEE, updateGraphicalView);
    connectLimbs(BP_RIGHT_KNEE, BP_RIGHT_ANKLE, updateGraphicalView);
    connectLimbs(BP_RIGHT_ANKLE, BP_RIGHT_HEEL, updateGraphicalView);
    connectLimbs(BP_RIGHT_ANKLE, BP_RIGHT_FOOT_INDEX, updateGraphicalView);
    connectLimbs(BP_RIGHT_HEEL, BP_RIGHT_FOOT_INDEX, updateGraphicalView);

    /* Draw dots on each detected joint */
    for (int i = 0; i <= BP_RIGHT_FOOT_INDEX; i ++) {
        QBrush brush;

        float x = xCoordinate[i];
        float y = yCoordinate[i];

        pen.setColor(DOT_COLOUR);

        if (x >= 0 && y >= 0) {
            if (updateGraphicalView)
                scenePointProjection->addEllipse(x, y, PEN_WIDTH, PEN_WIDTH, pen, brush);
            else
                scene->addEllipse(x, y, PEN_WIDTH, PEN_WIDTH, pen, brush);
        }
    }
}

void poseEstimation::connectLimbs(int limb1, int limb2, bool drawGraphicalViewLimbs)
{
    QPen pen;

    pen.setWidth(PEN_WIDTH);
    pen.setColor(LINE_COLOUR);

    if (std::isnan(xCoordinate[limb1]) == false && std::isnan(yCoordinate[limb1]) == false) {
        if (std::isnan(xCoordinate[limb2]) == false && std::isnan(yCoordinate[limb2]) == false) {
                    if (drawGraphicalViewLimbs)
                    uiPE->graphicsViewPointProjection->scene()->addLine(xCoordinate[limb1], yCoordinate[limb1], xCoordinate[limb2], yCoordinate[limb2], pen);
                    else
            		uiPE->graphicsView->scene()->addLine(xCoordinate[limb1], yCoordinate[limb1], xCoordinate[limb2], yCoordinate[limb2], pen);
    	}
    }
}

void poseEstimation::runInference(const QVector<float> &receivedTensor, int receivedStride, int receivedTimeElapsed, const cv::Mat &receivedMat)
{
    if (poseModelSet == BlazePose)
        outputTensor = sortTensorBlazePose(receivedTensor, receivedStride);
    else
        outputTensor = sortTensorMoveNet(receivedTensor, receivedStride);

    uiPE->labelInference->setText(TEXT_INFERENCE + QString("%1 ms").arg(receivedTimeElapsed));

    emit sendMatToView(receivedMat);
    if (continuousMode) {
        /* Stop Total FPS timer and display it to GUI, then restart timer before getting the next frame */
        timeTotalFps(false);
        timeTotalFps(true);
        emit getFrame();
    } else {
        setButtonState(true);
    }

    /* Draw onto image first then the graphical view */
    if (poseModelSet == BlazePose) {
        drawLimbsBlazePose(outputTensor, false);
        drawLimbsBlazePose(outputTensor, true);
    } else {
        drawLimbsMoveNet(outputTensor, false);
        drawLimbsMoveNet(outputTensor, true);
    }
}

void poseEstimation::stopContinuousMode()
{
    continuousMode = false;

    stopVideo();
    setButtonState(true);

    uiPE->labelInference->setText(TEXT_INFERENCE);
    uiPE->labelTotalFpsPose->setText(TEXT_TOTAL_FPS);
    uiPE->graphicsViewPointProjection->scene()->clear();

    if (inputModePE != videoMode)
        emit startVideo();
}

void poseEstimation::timeTotalFps(bool startingTimer)
{
    if (startingTimer) {
        /* Start the timer to measure Total FPS */
        startTime = std::chrono::high_resolution_clock::now();
    } else {
        /* Stop timer, calculate Total FPS and display to GUI */
        std::chrono::high_resolution_clock::time_point stopTime = std::chrono::high_resolution_clock::now();
        int timeElapsed = int(std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime).count());
        displayTotalFps(timeElapsed);
    }
}

void poseEstimation::displayTotalFps(int totalProcessTime)
{
    float totalFps = 1000.0 / totalProcessTime;

    uiPE->labelTotalFpsPose->setText(TEXT_TOTAL_FPS + QString::number(double(totalFps), 'f', 1));
}

void poseEstimation::triggerInference()
{
    if (inputModePE == imageMode) {
        continuousMode = false;

        stopVideo();
        setButtonState(false);

        emit getFrame();
    } else {
        if (buttonState) {
            continuousMode = true;

            timeTotalFps(true);
            setButtonState(false);
            stopVideo();
            emit getFrame();
        } else {
            continuousMode = false;

            setButtonState(true);

            if (inputModePE == videoMode) {
                stopVideo();
            } else {
                startVideo();
                uiPE->labelInference->setText(TEXT_INFERENCE);
                uiPE->labelTotalFpsPose->setText(TEXT_TOTAL_FPS);
                uiPE->graphicsViewPointProjection->scene()->clear();
            }
        }
    }
}

void poseEstimation::setCameraMode()
{
    inputModePE = cameraMode;

    uiPE->actionLoad_Camera->setEnabled(false);
    uiPE->actionLoad_File->setText(TEXT_LOAD_FILE);
}

void poseEstimation::setImageMode()
{
    inputModePE = imageMode;

    uiPE->actionLoad_Camera->setEnabled(true);
    uiPE->actionLoad_File->setText(TEXT_LOAD_NEW_FILE);
}

void poseEstimation::setVideoMode()
{
    inputModePE = videoMode;

    uiPE->actionLoad_Camera->setEnabled(true);
    uiPE->actionLoad_File->setText(TEXT_LOAD_NEW_FILE);
}

void poseEstimation::setFrameDims(int height, int width)
{
    frameHeight = height;
    frameWidth = width;
}
