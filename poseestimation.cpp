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

#include <QEventLoop>
#include <QGraphicsScene>
#include <QGraphicsTextItem>

#define DETECT_THRESHOLD 0.3

#define NOSE 0
#define LEFT_EYE 1
#define RIGHT_EYE 2
#define LEFT_EAR 3
#define RIGHT_EAR 4
#define LEFT_SHOULDER 5
#define RIGHT_SHOULDER 6
#define LEFT_ELBOW 7
#define RIGHT_ELBOW 8
#define LEFT_WRIST 9
#define RIGHT_WRIST 10
#define LEFT_HIP 11
#define RIGHT_HIP 12
#define LEFT_KNEE 13
#define RIGHT_KNEE 14
#define LEFT_ANKLE 15
#define RIGHT_ANKLE 16

#define PEN_WIDTH 2
#define DOT_COLOUR Qt::green
#define LINE_COLOUR Qt::red

poseEstimation::poseEstimation(Ui::MainWindow *ui)
{
    uiPE = ui;
    inputModePE = cameraMode;
    buttonState = true;

    uiPE->actionShopping_Basket->setDisabled(false);
    uiPE->actionObject_Detection->setDisabled(false);
    uiPE->actionPose_Estimation->setDisabled(true);
    uiPE->actionLoad_Camera->setDisabled(true);
    uiPE->actionLoad_File->setText(TEXT_LOAD_FILE);

    uiPE->labelInference->setText(TEXT_INFERENCE);
    uiPE->labelDemoMode->setText("Mode: Pose Estimation");
    uiPE->labelTotalFps->setText(TEXT_TOTAL_FPS);

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

QVector<float> poseEstimation::sortTensor(const QVector<float> receivedTensor)
{
    QVector<float> sortedTensor = QVector<float>();

    float nanValue = std::nanf("NAN");

    for(int i = 0; i < receivedTensor.size(); i += 3) {
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

void poseEstimation::drawLimbs(const QVector<float> &outputTensor, bool updateGraphicalView)
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

void poseEstimation::runInference(const QVector<float> &receivedTensor, int receivedTimeElapsed, const cv::Mat &receivedMat)
{
    outputTensor = sortTensor(receivedTensor);

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
    drawLimbs(outputTensor, false);
    drawLimbs(outputTensor, true);
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
