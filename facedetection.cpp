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

#include "facedetection.h"
#include "ui_mainwindow.h"

#include <QEventLoop>
#include <QGraphicsScene>
#include <QGraphicsTextItem>

#define FACE_LANDMARK_INPUT_SIZE 192.0

#define PEN_THICKNESS 2
#define DOT_COLOUR Qt::red
#define LINE_COLOUR Qt::green

#define DETECT_THRESHOLD_FACE 0.5

faceDetection::faceDetection(Ui::MainWindow *ui)
{
    uiFD = ui;
    inputModeFD = cameraMode;
    buttonState = true;

    utilFD = new edgeUtils();

    frameHeight = GRAPHICS_VIEW_HEIGHT;
    frameWidth = GRAPHICS_VIEW_WIDTH;

    uiFD->actionShopping_Basket->setDisabled(false);
    uiFD->actionObject_Detection->setDisabled(false);
    uiFD->actionPose_Estimation->setDisabled(false);
    uiFD->actionFace_Detection->setDisabled(true);

    uiFD->actionLoad_Camera->setDisabled(true);
    uiFD->actionLoad_File->setText(TEXT_LOAD_FILE);

    uiFD->labelInference->setText(TEXT_INFERENCE);
    uiFD->labelDemoMode->setText("Mode: Face Detection");
    uiFD->labelTotalFpsPose->setText(TEXT_TOTAL_FPS);

    uiFD->stackedWidgetLeft->setCurrentIndex(STACK_WIDGET_INDEX_FD);
    uiFD->stackedWidgetRight->setCurrentIndex(STACK_WIDGET_INDEX_FD);

    QGraphicsScene *scenePointProjection = new QGraphicsScene(this);
    uiFD->graphicsViewPointPlotFace->setScene(scenePointProjection);
}

void faceDetection::setButtonState(bool enable)
{
    buttonState = enable;

    if (buttonState) {
        uiFD->pushButtonStartStopFace->setText("Start\nInference");
        uiFD->pushButtonStartStopFace->setStyleSheet(BUTTON_BLUE);
    } else {
        uiFD->pushButtonStartStopFace->setText("Stop\nInference");
        uiFD->pushButtonStartStopFace->setStyleSheet(BUTTON_RED);
    }

    qApp->processEvents(QEventLoop::WaitForMoreEvents);
}

QVector<float> faceDetection::sortTensorFaceLandmark(const QVector<float> receivedTensor, int receivedStride)
{
    QVector<float> sortedTensor = QVector<float>();

    float nanValue = std::nanf("NAN");
    float confidenceValue = receivedTensor.at(receivedStride);

    /* Calculate confidence probability by applying sigmoid function */
    float confidenceLevel = 1 / (1 + exp(-confidenceValue));

    for (int i = 0; i < receivedStride; i += 3) {

        if (confidenceLevel > DETECT_THRESHOLD_FACE && confidenceLevel <= 1.0) {
            sortedTensor.push_back(receivedTensor.at(i + 1)); // y-coordinate
            sortedTensor.push_back(receivedTensor.at(i));     // x-coordinate
        } else {
            sortedTensor.push_back(nanValue);
            sortedTensor.push_back(nanValue);
        }
     }

    return sortedTensor;
}

void faceDetection::drawPointsFaceLandmark(const QVector<float> &outputTensor, bool updateGraphicalView)
{
    QPen pen;
    xCoordinate = QVector<float>();
    yCoordinate = QVector<float>();
    int displayWidth;
    int displayHeight;

    QGraphicsScene *scene = uiFD->graphicsView->scene();
    QGraphicsScene *scenePointProjection = uiFD->graphicsViewPointPlotFace->scene();

    /*
     * Indexing of the Face Landmark model can be found at
     * https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
     */
    QVector <int> pointsIndexFace = { 10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149,
                                     176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389,
                                     251, 284, 332, 297, 338, 10 };
    QVector <int> pointsIndexInnerLip = { 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 95, 88, 178, 87, 14,
                                          317, 402, 318, 324, 308 };
    QVector <int> pointsIndexOuterLip = { 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291 };
    QVector <int> pointsIndexLEye = { 362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380,
                                      381, 382, 362 };
    QVector <int> pointsIndexLEyebrowBottom = { 285, 295, 282, 283, 276 };
    QVector <int> pointsIndexLEyebrowTop = { 336, 296, 334, 293, 300 };
    QVector <int> pointsIndexREye = { 133, 173, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154,
                                      155, 133 };
    QVector <int> pointsIndexREyebrowBottom = { 55, 65, 52, 53, 46 };
    QVector <int> pointsIndexREyebrowTop = { 107, 66, 105, 63, 70 };

    pen.setWidth(PEN_THICKNESS);

    if (updateGraphicalView) {
        /* Scale the dimensions down by 2 */
        displayHeight = frameHeight / 2;
        displayWidth = frameWidth / 2;

        scenePointProjection->clear();
    } else {
        displayHeight = frameHeight;
        displayWidth = frameWidth;
    }

    float widthMultiplier = float(displayWidth) / FACE_LANDMARK_INPUT_SIZE;
    float heightMultiplier = float(displayHeight) / FACE_LANDMARK_INPUT_SIZE;

    /* Save x and y coordinates in separate vectors */
    for (int i = 0; i < outputTensor.size(); i += 2) {
        float y = outputTensor[i] * heightMultiplier;
        float x = outputTensor[i + 1] * widthMultiplier;

        xCoordinate.push_back(x);
        yCoordinate.push_back(y);
    }

    QList<QVector<int>> *faceParts = new QList<QVector<int>>();

    faceParts->push_back(pointsIndexFace);
    faceParts->push_back(pointsIndexInnerLip);
    faceParts->push_back(pointsIndexOuterLip);
    faceParts->push_back(pointsIndexLEye);
    faceParts->push_back(pointsIndexLEyebrowBottom);
    faceParts->push_back(pointsIndexLEyebrowTop);
    faceParts->push_back(pointsIndexREye);
    faceParts->push_back(pointsIndexREyebrowBottom);
    faceParts->push_back(pointsIndexREyebrowTop);

    /* Draw lines on image frame and dots on the 2D Point Projection */
    if (!updateGraphicalView) {
        foreach (const QVector<int> &faceIdentified, *faceParts) {
            for (int i = 0; (i + 1) < faceIdentified.size(); i++)
                connectLandmarks(faceIdentified.at(i), faceIdentified.at(i + 1), updateGraphicalView);
        }
    } else {
        /* Draw dots on each detected landmark */
        for (int i = 0; i < (outputTensor.size() / 2); i ++) {
            QBrush brush;

            float x = xCoordinate[i];
            float y = yCoordinate[i];

            pen.setColor(DOT_COLOUR);

            if (x >= 0 && y >= 0) {
                if (updateGraphicalView)
                    scenePointProjection->addEllipse(x, y, PEN_THICKNESS, PEN_THICKNESS, pen, brush);
                else
                    scene->addEllipse(x, y, PEN_THICKNESS, PEN_THICKNESS, pen, brush);
            }
        }
    }
}

void faceDetection::connectLandmarks(int landmark1, int landmark2, bool drawGraphicalViewLandmarks)
{
    QPen pen;
    bool nanLandmark1;
    bool nanLandmark2;

    pen.setWidth(PEN_THICKNESS);
    pen.setColor(LINE_COLOUR);

    nanLandmark1 = std::isnan(xCoordinate[landmark1]) && std::isnan(yCoordinate[landmark1]);
    nanLandmark2 = std::isnan(xCoordinate[landmark2]) && std::isnan(yCoordinate[landmark2]);

    if (!nanLandmark1 || !nanLandmark2) {
        QLine *lineToDraw = new QLine(xCoordinate[landmark1], yCoordinate[landmark1], xCoordinate[landmark2], yCoordinate[landmark2]);

        if (drawGraphicalViewLandmarks)
            uiFD->graphicsViewPointPlotFace->scene()->addLine(*lineToDraw, pen);
        else
            uiFD->graphicsView->scene()->addLine(*lineToDraw, pen);
    }
}

void faceDetection::runInference(const QVector<float> &receivedTensor, int receivedStride, int receivedTimeElapsed, const cv::Mat &receivedMat)
{
    float totalFps;

    outputTensor = sortTensorFaceLandmark(receivedTensor, receivedStride);

    uiFD->labelInference->setText(TEXT_INFERENCE + QString("%1 ms").arg(receivedTimeElapsed));

    emit sendMatToView(receivedMat);

    if (continuousMode) {
        /* Stop Total FPS timer and display it to GUI, then restart timer before getting the next frame */
        utilFD->timeTotalFps(false);
        totalFps = utilFD->calculateTotalFps();
        uiFD->labelTotalFpsFace->setText(TEXT_TOTAL_FPS + QString::number(double(totalFps), 'f', 1));
        utilFD->timeTotalFps(true);

        emit getFrame();
    } else {
        setButtonState(true);
    }

    /* Draw onto image first then the graphical view */
    drawPointsFaceLandmark(outputTensor, false);
    drawPointsFaceLandmark(outputTensor, true);
}

void faceDetection::stopContinuousMode()
{
    continuousMode = false;

    stopVideo();
    setButtonState(true);

    uiFD->labelInference->setText(TEXT_INFERENCE);
    uiFD->labelTotalFpsFace->setText(TEXT_TOTAL_FPS);
    uiFD->graphicsViewPointPlotFace->scene()->clear();

    if (inputModeFD != videoMode)
        emit startVideo();
}

void faceDetection::triggerInference()
{
    if (inputModeFD == imageMode) {
        continuousMode = false;

        stopVideo();
        setButtonState(false);

        emit getFrame();
    } else {
        if (buttonState) {
            continuousMode = true;

            utilFD->timeTotalFps(true);
            setButtonState(false);
            stopVideo();
            emit getFrame();
        } else {
            continuousMode = false;

            setButtonState(true);

            if (inputModeFD == videoMode) {
                stopVideo();
            } else {
                startVideo();
                uiFD->labelInference->setText(TEXT_INFERENCE);
                uiFD->labelTotalFpsFace->setText(TEXT_TOTAL_FPS);
                uiFD->graphicsViewPointPlotFace->scene()->clear();
            }
        }
    }
}

void faceDetection::setCameraMode()
{
    inputModeFD = cameraMode;

    uiFD->actionLoad_Camera->setEnabled(false);
    uiFD->actionLoad_File->setText(TEXT_LOAD_FILE);
}

void faceDetection::setImageMode()
{
    inputModeFD = imageMode;

    uiFD->actionLoad_Camera->setEnabled(true);
    uiFD->actionLoad_File->setText(TEXT_LOAD_NEW_FILE);
}

void faceDetection::setVideoMode()
{
    inputModeFD = videoMode;

    uiFD->actionLoad_Camera->setEnabled(true);
    uiFD->actionLoad_File->setText(TEXT_LOAD_NEW_FILE);
}

void faceDetection::setFrameDims(int height, int width)
{
    frameHeight = height;
    frameWidth = width;
}
