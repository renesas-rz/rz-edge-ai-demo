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

#include <QEventLoop>
#include <QGraphicsScene>
#include <QGraphicsTextItem>

#define BLAZE_POSE_INPUT_SIZE 256.0
#define HAND_POSE_INPUT_SIZE 224.0

#define HAND_POSE_CONFIDENCE_INDEX 63

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

enum HandPosePoints { HP_WRIST, HP_THUMB_CMC, HP_THUMB_MCP, HP_THUMB_IP, HP_THUMB_TIP, HP_INDEX_FINGER_MCP,
                      HP_INDEX_FINGER_PIP, HP_INDEX_FINGER_DIP, HP_INDEX_FINGER_TIP, HP_MIDDLE_FINGER_MCP,
                      HP_MIDDLE_FINGER_PIP, HP_MIDDLE_FINGER_DIP, HP_MIDDLE_FINGER_TIP, HP_RING_FINGER_MCP,
                      HP_RING_FINGER_PIP, HP_RING_FINGER_DIP, HP_RING_FINGER_TIP, HP_PINKY_MCP, HP_PINKY_PIP,
                      HP_PINKY_DIP, HP_PINKY_TIP };

#define PEN_WIDTH 2
#define PEN_WIDTH_HAND_POSE 3

poseEstimation::poseEstimation(Ui::MainWindow *ui, QString modelPath, QString inferenceEngine, bool cameraConnect)
{
    QString modelName;

    uiPE = ui;
    inputModePE = cameraMode;
    buttonState = true;
    camConnect = cameraConnect;

    utilPE = new edgeUtils();

    if (modelPath.contains(IDENTIFIER_MOVE_NET))
        poseModelSet = MoveNet;
    else if (modelPath.contains(IDENTIFIER_HAND_POSE))
        poseModelSet = HandPose;
    else
        poseModelSet = BlazePose;

    modelName = modelPath.section('/', -1);
    frameHeight = GRAPHICS_VIEW_HEIGHT;
    frameWidth = GRAPHICS_VIEW_WIDTH;

    uiPE->actionShopping_Basket->setDisabled(false);
    uiPE->actionObject_Detection->setDisabled(false);
    uiPE->actionPose_Estimation->setDisabled(true);
    uiPE->actionFace_Detection->setDisabled(false);
    uiPE->actionLoad_File->setText(TEXT_LOAD_FILE);
    uiPE->actionLoad_Camera->setDisabled(true);

    uiPE->labelAIModelFilenamePE->setText(modelName);
    uiPE->labelInferenceEnginePE->setText(inferenceEngine);
    uiPE->labelInferenceTimePE->setText(TEXT_INFERENCE);
    uiPE->labelDemoMode->setText("Mode: Pose Estimation");
    uiPE->labelTotalFpsPose->setText(TEXT_TOTAL_FPS);

    uiPE->stackedWidgetLeft->setCurrentIndex(STACK_WIDGET_INDEX_PE);
    uiPE->stackedWidgetRight->setCurrentIndex(STACK_WIDGET_INDEX_PE);

    QGraphicsScene *scenePointProjection = new QGraphicsScene(this);
    uiPE->graphicsViewPointProjection->setScene(scenePointProjection);
}

void poseEstimation::setButtonState(bool enable)
{
    buttonState = enable;

    if (buttonState) {
        uiPE->pushButtonStartStopPose->setText("Start\nInference");
        uiPE->pushButtonStartStopPose->setStyleSheet(BUTTON_BLUE);
    } else {
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
        float confidenceLevel = edgeUtils::calculateSigmoid(receivedTensor.at(i + 4));

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

QVector<float> poseEstimation::sortTensorHandPose(const QVector<float> receivedTensor, int receivedStride)
{
    QVector<float> sortedTensor = QVector<float>();

    float nanValue = std::nanf("NAN");
    float confidenceValue = receivedTensor.at(HAND_POSE_CONFIDENCE_INDEX);

    for(int i = 0; i < receivedStride; i += 3) {

        if (confidenceValue > 0.5 && confidenceValue <= 1.0) {
            sortedTensor.push_back(receivedTensor.at(i + 1)); // y-coordinate
            sortedTensor.push_back(receivedTensor.at(i));     // x-coordinate
        } else {
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

        pen.setColor(DOT_GREEN);

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

        pen.setColor(DOT_GREEN);

        if (x >= 0 && y >= 0) {
            if (updateGraphicalView)
                scenePointProjection->addEllipse(x, y, PEN_WIDTH, PEN_WIDTH, pen, brush);
            else
                scene->addEllipse(x, y, PEN_WIDTH, PEN_WIDTH, pen, brush);
        }
    }
}

void poseEstimation::drawLimbsHandPose(const QVector<float> &outputTensor, bool updateGraphicalView)
{
    QPen pen;
    xCoordinate = QVector<float>();
    yCoordinate = QVector<float>();
    int displayWidth;
    int displayHeight;

    QGraphicsScene *scene = uiPE->graphicsView->scene();
    QGraphicsScene *scenePointProjection = uiPE->graphicsViewPointProjection->scene();

    pen.setWidth(PEN_WIDTH_HAND_POSE);

    if (updateGraphicalView) {
        /* Scale the dimensions down by 2 */
        displayHeight = frameHeight / 2;
        displayWidth = frameWidth / 2;

        scenePointProjection->clear();
    } else {
        displayHeight = frameHeight;
        displayWidth = frameWidth;
    }

    float widthMultiplier = float(displayWidth) / HAND_POSE_INPUT_SIZE;
    float heightMultiplier = float(displayHeight) / HAND_POSE_INPUT_SIZE;

    /* Save x and y coordinates in separate vectors */
    for (int i = 0; i < outputTensor.size(); i += 2) {
        float y = outputTensor[i] * heightMultiplier;
        float x = outputTensor[i + 1] * widthMultiplier;

        xCoordinate.push_back(x);
        yCoordinate.push_back(y);
    }

    /* Draw lines between the hand-knuckle coordinates */
    connectLimbs(HP_WRIST, HP_THUMB_CMC, updateGraphicalView);
    connectLimbs(HP_THUMB_CMC, HP_THUMB_MCP, updateGraphicalView);
    connectLimbs(HP_THUMB_MCP, HP_THUMB_IP, updateGraphicalView);
    connectLimbs(HP_THUMB_IP, HP_THUMB_TIP, updateGraphicalView);
    connectLimbs(HP_WRIST, HP_INDEX_FINGER_MCP, updateGraphicalView);
    connectLimbs(HP_INDEX_FINGER_MCP, HP_INDEX_FINGER_PIP, updateGraphicalView);
    connectLimbs(HP_INDEX_FINGER_PIP, HP_INDEX_FINGER_DIP, updateGraphicalView);
    connectLimbs(HP_INDEX_FINGER_DIP, HP_INDEX_FINGER_TIP, updateGraphicalView);
    connectLimbs(HP_INDEX_FINGER_MCP, HP_MIDDLE_FINGER_MCP, updateGraphicalView);
    connectLimbs(HP_MIDDLE_FINGER_MCP, HP_MIDDLE_FINGER_PIP, updateGraphicalView);
    connectLimbs(HP_MIDDLE_FINGER_PIP, HP_MIDDLE_FINGER_DIP, updateGraphicalView);
    connectLimbs(HP_MIDDLE_FINGER_DIP, HP_MIDDLE_FINGER_TIP, updateGraphicalView);
    connectLimbs(HP_MIDDLE_FINGER_MCP, HP_RING_FINGER_MCP, updateGraphicalView);
    connectLimbs(HP_RING_FINGER_MCP, HP_RING_FINGER_PIP, updateGraphicalView);
    connectLimbs(HP_RING_FINGER_PIP, HP_RING_FINGER_DIP, updateGraphicalView);
    connectLimbs(HP_RING_FINGER_DIP, HP_RING_FINGER_TIP, updateGraphicalView);
    connectLimbs(HP_RING_FINGER_MCP, HP_PINKY_MCP, updateGraphicalView);
    connectLimbs(HP_PINKY_MCP, HP_PINKY_PIP, updateGraphicalView);
    connectLimbs(HP_PINKY_PIP, HP_PINKY_DIP, updateGraphicalView);
    connectLimbs(HP_PINKY_DIP, HP_PINKY_TIP, updateGraphicalView);
    connectLimbs(HP_PINKY_MCP, HP_WRIST, updateGraphicalView);

    /* Draw dots on each detected hand-knuckle coordinates */
    for (int i = 0; i <= HP_PINKY_TIP; i ++) {
        QBrush brush;

        float x = xCoordinate[i];
        float y = yCoordinate[i];

        pen.setColor(DOT_GREEN);

        if (x >= 0 && y >= 0) {
            if (updateGraphicalView)
                scenePointProjection->addEllipse(x, y, PEN_WIDTH_HAND_POSE, PEN_WIDTH_HAND_POSE, pen, brush);
            else
                scene->addEllipse(x, y, PEN_WIDTH_HAND_POSE, PEN_WIDTH_HAND_POSE, pen, brush);
        }
    }
}

void poseEstimation::connectLimbs(int limb1, int limb2, bool drawGraphicalViewLimbs)
{
    QPen pen;
    bool nanLimb1;
    bool nanLimb2;

    if (poseModelSet == HandPose) {
        pen.setWidth(PEN_WIDTH_HAND_POSE);
        pen.setColor(LINE_BLUE);
    } else {
        pen.setWidth(PEN_WIDTH);
        pen.setColor(LINE_RED);
    }

    nanLimb1 = !std::isnan(xCoordinate[limb1]) || !std::isnan(yCoordinate[limb1]);
    nanLimb2 = !std::isnan(xCoordinate[limb2]) || !std::isnan(yCoordinate[limb2]);

    if (nanLimb1 && nanLimb2) {
        QLine *lineToDraw = new QLine(xCoordinate[limb1], yCoordinate[limb1], xCoordinate[limb2], yCoordinate[limb2]);

        if (drawGraphicalViewLimbs)
            uiPE->graphicsViewPointProjection->scene()->addLine(*lineToDraw, pen);
        else
            uiPE->graphicsView->scene()->addLine(*lineToDraw, pen);
    }
}

void poseEstimation::runInference(const QVector<float> &receivedTensor, int receivedStride, int receivedTimeElapsed, const cv::Mat &receivedMat)
{
    float totalFps;

    if (poseModelSet == MoveNet)
        outputTensor = sortTensorMoveNet(receivedTensor, receivedStride);
    else if (poseModelSet == HandPose)
        outputTensor = sortTensorHandPose(receivedTensor, receivedStride);
    else
        outputTensor = sortTensorBlazePose(receivedTensor, receivedStride);

    uiPE->labelInferenceTimePE->setText(TEXT_INFERENCE + QString("%1 ms").arg(receivedTimeElapsed));

    emit sendMatToView(receivedMat);
    if (continuousMode) {
        /* Stop Total FPS timer and display it to GUI, then restart timer before getting the next frame */
        utilPE->timeTotalFps(false);
        totalFps = utilPE->calculateTotalFps();
        uiPE->labelTotalFpsPose->setText(TEXT_TOTAL_FPS + QString::number(double(totalFps), 'f', 1));
        utilPE->timeTotalFps(true);

        emit getFrame();
    } else {
        setButtonState(true);
    }

    /* Draw onto image first then the graphical view */
    if (poseModelSet == MoveNet) {
        drawLimbsMoveNet(outputTensor, false);
        drawLimbsMoveNet(outputTensor, true);
    } else if (poseModelSet == HandPose) {
        drawLimbsHandPose(outputTensor, false);
        drawLimbsHandPose(outputTensor, true);
    } else {
        drawLimbsBlazePose(outputTensor, false);
        drawLimbsBlazePose(outputTensor, true);
    }
}

void poseEstimation::stopContinuousMode()
{
    continuousMode = false;

    stopVideo();
    setButtonState(true);

    uiPE->labelInferenceTimePE->setText(TEXT_INFERENCE);
    uiPE->labelTotalFpsPose->setText(TEXT_TOTAL_FPS);
    uiPE->graphicsViewPointProjection->scene()->clear();

    if (inputModePE != videoMode)
        emit startVideo();
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

            utilPE->timeTotalFps(true);
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
                uiPE->labelInferenceTimePE->setText(TEXT_INFERENCE);
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

    uiPE->actionLoad_Camera->setEnabled(camConnect);
    uiPE->actionLoad_File->setText(TEXT_LOAD_NEW_FILE);
}

void poseEstimation::setVideoMode()
{
    inputModePE = videoMode;

    uiPE->actionLoad_Camera->setEnabled(camConnect);
    uiPE->actionLoad_File->setText(TEXT_LOAD_NEW_FILE);
}

void poseEstimation::setFrameDims(int height, int width)
{
    frameHeight = height;
    frameWidth = width;
}
