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

#include <opencv2/imgproc/imgproc.hpp>

#include <QEventLoop>
#include <QGraphicsScene>
#include <QGraphicsTextItem>

#define FACE_DETECTION_BOX_INDEX 4
#define FACE_DETECTION_INPUT_SIZE 128.0
#define FACE_DETECTION_OUTPUT_INDEX 16
#define FACE_LANDMARK_INPUT_SIZE 192.0
#define IRIS_LANDMARK_INPUT_SIZE 64.0
#define IRIS_LANDMARK_IRIS_OUTPUT_INDEX 10

#define PEN_THICKNESS 2

#define DETECT_THRESHOLD_FACE 0.5
#define ANCHOR_CENTER 0.5
#define DETECT_BOUNDING_BOX_INCREASE 0.1 //Needed to ensure crop contains entire face

struct {
    float x;
    float y;
    float width;
    float height;
} eyeLeft, eyeRight;

faceDetection::faceDetection(Ui::MainWindow *ui, QString inferenceEngine, DetectMode detectModeToUse)
{
    QPixmap irisDiagram(IRIS_DIAGRAM_PATH);

    uiFD = ui;
    inputModeFD = cameraMode;
    faceModel = faceDetect;
    detectMode = detectModeToUse;
    buttonState = true;
    faceVisible = false;

    utilFD = new edgeUtils();

    frameHeight = GRAPHICS_VIEW_HEIGHT;
    frameWidth = GRAPHICS_VIEW_WIDTH;

    uiFD->actionShopping_Basket->setDisabled(false);
    uiFD->actionObject_Detection->setDisabled(false);
    uiFD->actionPose_Estimation->setDisabled(false);
    uiFD->actionFace_Detection->setDisabled(true);

    uiFD->actionLoad_Camera->setDisabled(true);
    uiFD->actionLoad_File->setText(TEXT_LOAD_FILE);

    uiFD->labelIrisDiagram->setPixmap(irisDiagram);
    uiFD->labelAIModelFilenameFD->setText(TEXT_FACE_MODEL);
    uiFD->labelInferenceEngineFD->setText(inferenceEngine);
    uiFD->labelDemoMode->setText("Mode: Face Detection");
    uiFD->labelTotalFpsPose->setText(TEXT_TOTAL_FPS);

    uiFD->stackedWidgetLeft->setCurrentIndex(STACK_WIDGET_INDEX_FD);
    uiFD->stackedWidgetRight->setCurrentIndex(STACK_WIDGET_INDEX_FD);

    QGraphicsScene *scenePointProjection = new QGraphicsScene(this);
    uiFD->graphicsViewPointPlotFace->setScene(scenePointProjection);

    if (detectMode == irisMode)
        detectIrisMode();
    else
        detectFaceMode();
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
            sortedTensor.push_back(receivedTensor.at(i));     // x-coordinate
            sortedTensor.push_back(receivedTensor.at(i + 1)); // y-coordinate
        } else {
            sortedTensor.push_back(nanValue);
            sortedTensor.push_back(nanValue);
        }
     }

    return sortedTensor;
}

QVector<float> faceDetection::sortTensorIrisLandmark(const QVector<float> receivedTensor, int receivedStride)
{
    QVector<float> sortedTensorIris = QVector<float>();

    float nanValueIris = std::nanf("NAN");

    for (int i = receivedStride; i < receivedTensor.size(); i += 3) {
        if (receivedTensor.at(i) >= 0 && receivedTensor.at(i + 1) >= 0) {
            sortedTensorIris.push_back(receivedTensor.at(i));     // x-coordinate
            sortedTensorIris.push_back(receivedTensor.at(i + 1)); // y-coordinate
        } else {
            sortedTensorIris.push_back(nanValueIris);
            sortedTensorIris.push_back(nanValueIris);
        }
    }

    return sortedTensorIris;
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

    int pointPlotHeight = uiFD->graphicsViewPointPlotFace->height();
    int pointPlotWidth = uiFD->graphicsViewPointPlotFace->width();

    if (updateGraphicalView) {
        /*
         * Scale the dimensions down by 1.5 when the face is larger than the
         * 2D Point Projection
         */
        if (faceHeight > pointPlotHeight || faceWidth > pointPlotWidth) {
            displayHeight = faceHeight / 1.5;
            displayWidth = faceWidth / 1.5;
        } else {
            displayHeight = faceHeight;
            displayWidth = faceWidth;
        }

        scenePointProjection->clear();
    } else {
        displayHeight = faceHeight;
        displayWidth = faceWidth;
    }

    float widthMultiplier = float(displayWidth) / FACE_LANDMARK_INPUT_SIZE;
    float heightMultiplier = float(displayHeight) / FACE_LANDMARK_INPUT_SIZE;

    /* Save x and y coordinates in separate vectors */
    for (int i = 0; i < outputTensor.size(); i += 2) {
        float x = outputTensor[i] * widthMultiplier;
        float y = outputTensor[i + 1] * heightMultiplier;

        /*
         * Ensure coordinates drawn onto image frame are relative to the cropping
         * region of the original image
         */
        if (!updateGraphicalView) {
            y += faceTopLeftY;
            x += faceTopLeftX;
        }

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

            pen.setColor(DOT_RED);

            if (x >= 0 && y >= 0) {
                if (updateGraphicalView)
                    scenePointProjection->addEllipse(x, y, PEN_THICKNESS, PEN_THICKNESS, pen, brush);
                else
                    scene->addEllipse(x, y, PEN_THICKNESS, PEN_THICKNESS, pen, brush);
            }
        }
    }
}

void faceDetection::drawPointsIrisLandmark(const QVector<float> &outputTensor, bool drawLeftEye)
{
    QBrush brush = QBrush(Qt::white, Qt::Dense6Pattern);
    QPen pen;
    xCoordinate = QVector<float>();
    yCoordinate = QVector<float>();
    float displayWidth;
    float displayHeight;

    pen.setColor(Qt::yellow);
    pen.setWidth(PEN_THICKNESS);

    if (drawLeftEye) {
        displayWidth = eyeLeft.width / IRIS_LANDMARK_INPUT_SIZE;
        displayHeight = eyeLeft.height / IRIS_LANDMARK_INPUT_SIZE;
    } else {
        displayWidth = eyeRight.width / IRIS_LANDMARK_INPUT_SIZE;
        displayHeight = eyeRight.height / IRIS_LANDMARK_INPUT_SIZE;
    }

    /*
     * Save x and y coordinates in separate vectors and ensure that they
     * are relative to the cropping region of the original image
     */
    for (int i = 0; i < IRIS_LANDMARK_IRIS_OUTPUT_INDEX; i += 2) {
        float xIrisPosition;
        float yIrisPosition;

        if (drawLeftEye) {
            xIrisPosition = faceTopLeftX + eyeLeft.x;
            yIrisPosition = faceTopLeftY + eyeLeft.y;
        } else {
            xIrisPosition = faceTopLeftX + eyeRight.x;
            yIrisPosition = faceTopLeftY + eyeRight.y;
        }

        xIrisPosition += outputTensor[i] * displayWidth;
        yIrisPosition += outputTensor[i + 1] * displayHeight;

        xCoordinate.push_back(xIrisPosition);
        yCoordinate.push_back(yIrisPosition);
    }

    uiFD->graphicsView->scene()->addEllipse(xCoordinate.at(3), yCoordinate.at(2), (xCoordinate.at(1) - xCoordinate.at(3)), (yCoordinate.at(4) - yCoordinate.at(2)), pen, brush);
}

void faceDetection::connectLandmarks(int landmark1, int landmark2, bool drawGraphicalViewLandmarks)
{
    QPen pen;
    bool nanLandmark1;
    bool nanLandmark2;

    pen.setWidth(PEN_THICKNESS);
    pen.setColor(LINE_GREEN);

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

void faceDetection::runInference(const QVector<float> &receivedTensor, int receivedStride, int receivedTimeElapsed)
{
    float totalFps;

    if (detectMode == irisMode)
        outputTensor = sortTensorIrisLandmark(receivedTensor, receivedStride);
    else
        outputTensor = sortTensorFaceLandmark(receivedTensor, receivedStride);

    uiFD->labelInferenceTimeFaceDetection->setText(TEXT_INFERENCE_FACE_DETECTION + QString("%1 ms").arg(timeElaspedFaceDetection));

    if (detectMode == irisMode) {
        uiFD->labelInferenceTimeFaceLandmark->setText(TEXT_INFERENCE_FACE_LANDMARK + QString("%1 ms").arg(timeElaspedFaceLandmark));
        uiFD->labelInferenceTimeIrisLandmark->setText(TEXT_INFERENCE_IRIS_LANDMARK + QString("%1 ms").arg(timeElaspedIrisLeft) + QString(" + %1ms").arg(receivedTimeElapsed));
    } else {
        uiFD->labelInferenceTimeFaceLandmark->setText(TEXT_INFERENCE_FACE_LANDMARK + QString("%1 ms").arg(receivedTimeElapsed));
    }

    emit sendMatToView(resizedMat);

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

    if (detectMode == irisMode) {
        /* Draw left iris first and then right iris */
        drawPointsIrisLandmark(leftEyeTensor, true);
        drawPointsIrisLandmark(outputTensor, false);
    } else {
        /* Draw onto image first then the graphical view */
        drawPointsFaceLandmark(outputTensor, false);
        drawPointsFaceLandmark(outputTensor, true);
    }
}

void faceDetection::processFace(const cv::Mat &matToProcess)
{
    cv::Mat croppedFaceMat;
    bool detectIris;

    if (detectMode == irisMode)
        detectIris = true;
    else
        detectIris = false;

    faceModel = faceDetect;

    /* Resize cv::Mat and run inference using Face Detection model */
    cv::resize(matToProcess, resizedMat, cv::Size(frameWidth, frameHeight));

    emit sendMatForInference(resizedMat, faceModel, detectIris);

    faceModel = faceLandmark;

    /*
     * Crop cv::Mat using coordinates provided by Face Detection and
     * run inference using Face Landmark model
     */
    cv::Rect cropRegionFace(faceTopLeftX, faceTopLeftY, faceWidth, faceHeight);

    croppedFaceMat = resizedMat(cropRegionFace);

    emit sendMatForInference(croppedFaceMat, faceModel, detectIris);

    if (detectMode == irisMode)
        processIris(croppedFaceMat, detectIris);
}

void faceDetection::processIris(const cv::Mat &croppedFaceMat, bool detectIris)
{
    if (faceVisible) {
        cv::Mat croppedEyeMat;
        float irisScaleWidth = croppedFaceMat.cols / FACE_LANDMARK_INPUT_SIZE;
        float irisScaleHeight = croppedFaceMat.rows / FACE_LANDMARK_INPUT_SIZE;

        /* Process left eye */
        faceModel = irisLandmarkL;

        eyeLeft.x = eyeCropCoords.at(0) * irisScaleWidth;
        eyeLeft.y = eyeCropCoords.at(1) * irisScaleHeight;
        eyeLeft.width = (eyeCropCoords.at(2) * irisScaleWidth) - eyeLeft.x;
        eyeLeft.height = (eyeCropCoords.at(3) * irisScaleHeight) - eyeLeft.y;

        /*
         * Crop cv::Mat using coordinates provided by Face Landmark and
         * run inference on left eye using Iris Landmark model
         */
        cv::Rect cropRegionEyeL(eyeLeft.x, eyeLeft.y, eyeLeft.width, eyeLeft.height);

        croppedEyeMat = croppedFaceMat(cropRegionEyeL);

        emit sendMatForInference(croppedEyeMat, faceModel, detectIris);

        /* Process right eye */
        faceModel = irisLandmarkR;

        eyeRight.x = eyeCropCoords.at(4) * irisScaleWidth;
        eyeRight.y = eyeCropCoords.at(5) * irisScaleHeight;
        eyeRight.width = (eyeCropCoords.at(6) * irisScaleWidth) - eyeRight.x;
        eyeRight.height = (eyeCropCoords.at(7) * irisScaleHeight) - eyeRight.y;

        /*
         * Crop cv::Mat using coordinates provided by Face Landmark and
         * run inference on right eye using Iris Landmark model
         */
        cv::Rect cropRegionEyeR(eyeRight.x, eyeRight.y, eyeRight.width, eyeRight.height);

        croppedEyeMat = croppedFaceMat(cropRegionEyeR);

        emit sendMatForInference(croppedEyeMat, faceModel, detectIris);
    } else {
        int totalIrisModeFps;

        uiFD->labelInferenceTimeFaceDetection->setText(TEXT_INFERENCE_FACE_DETECTION + QString("%1 ms").arg(timeElaspedFaceDetection));
        uiFD->labelInferenceTimeFaceLandmark->setText(TEXT_INFERENCE_FACE_LANDMARK + QString("%1 ms").arg(timeElaspedFaceLandmark));
        uiFD->labelInferenceTimeIrisLandmark->setText(TEXT_INFERENCE_IRIS_LANDMARK);

        emit displayFrame();

        if (continuousMode) {
            /* Stop Total FPS timer and display it to GUI, then restart timer before getting the next frame */
            utilFD->timeTotalFps(false);
            totalIrisModeFps = utilFD->calculateTotalFps();
            uiFD->labelTotalFpsFace->setText(TEXT_TOTAL_FPS + QString::number(double(totalIrisModeFps), 'f', 1));
            utilFD->timeTotalFps(true);

            emit getFrame();
        } else {
            setButtonState(true);
        }
    }
}

void faceDetection::cropImageFace(const QVector<float> &faceDetectOutputTensor, int receivedStride, int receivedTimeElapsed, const cv::Mat &receivedMat)
{
    QVector<QPair<float, float>> anchorCoords;
    QVector<float> coordinatesTensor;
    QVector<float> confidenceTensor;
    QVector<float> croppedFaceDims;
    float inputImageHeight = receivedMat.rows;
    float inputImageWidth = receivedMat.cols;
    float faceDetectScaleHeight = inputImageHeight / FACE_DETECTION_INPUT_SIZE;
    float faceDetectScaleWidth = inputImageWidth / FACE_DETECTION_INPUT_SIZE;

    timeElaspedFaceDetection = receivedTimeElapsed;

    anchorCoords = generateAnchorCoords(inputImageHeight, inputImageWidth);

    for (int j = receivedStride; j < faceDetectOutputTensor.size(); j ++) {
        int iteration = j - receivedStride;

        /* Calculate confidence probability by applying sigmoid function */
        float confidenceLevel = 1 / (1 + exp(-faceDetectOutputTensor.at(j)));

        if (confidenceLevel > DETECT_THRESHOLD_FACE && confidenceLevel <= 1.0) {
            /*
             * BlazeFace outputs the x and y coordinates as offsets from an anchor point, so
             * the anchor coordinates must be added to the values
             */
            float yCenter = faceDetectOutputTensor.at(iteration * FACE_DETECTION_OUTPUT_INDEX) + anchorCoords.at(iteration).second;
            float xCenter = faceDetectOutputTensor.at(iteration * FACE_DETECTION_OUTPUT_INDEX + 1) + anchorCoords.at(iteration).first;
            float height = faceDetectOutputTensor.at(iteration * FACE_DETECTION_OUTPUT_INDEX + 2);
            float width = faceDetectOutputTensor.at(iteration * FACE_DETECTION_OUTPUT_INDEX + 3);

            /*
             * Scale coordinates to the input image and provide the top left coordinates
             * along with the height and width of the box
             */
            float xTopLeft = (xCenter - 2 * width);
            float yTopLeft = (yCenter - 2 * height);
            float scaledWidth = width * faceDetectScaleWidth;
            float scaledHeight = height * faceDetectScaleHeight;

            coordinatesTensor.push_back(xTopLeft);
            coordinatesTensor.push_back(yTopLeft);
            coordinatesTensor.push_back(scaledWidth);
            coordinatesTensor.push_back(scaledHeight);
            confidenceTensor.push_back(confidenceLevel);
        }
    }

    croppedFaceDims = sortBoundingBoxes(confidenceTensor, coordinatesTensor);

    setFaceCropDims(croppedFaceDims);
}

QVector<QPair<float, float>> faceDetection::generateAnchorCoords(int inputHeight, int inputWidth)
{
    /* BlazeFace uses two Conv layers (16x16, 8x8) for anchor computation */
    QVector<int> anchorGridDims = {16, 8};
    QVector<int> anchorTotalPoints = {2, 6};
    QPair<float, float> anchor;
    QVector<QPair<float, float>> anchorList;

    /* Get x and y anchor coordinates and store the points to a QVector */
    for (int i = 0; i < anchorGridDims.size(); i++) {
        int gridSize = anchorGridDims.at(i);
        float strideHeight = inputHeight / gridSize;
        float strideWidth = inputWidth / gridSize;
        int anchorAmount = anchorTotalPoints.at(i);

        for (int y = 0; y < gridSize; y++) {
            anchor.second = strideHeight * (y + ANCHOR_CENTER);

            for (int x = 0; x < gridSize; x++) {
                anchor.first = strideWidth * (x + ANCHOR_CENTER);

                for (int n = 0; n < anchorAmount; n++)
                    anchorList.push_back(anchor);
            }
        }
    }

    return anchorList;
}

QVector<float> faceDetection::sortBoundingBoxes(const QVector<float> receivedConfidenceTensor, const QVector<float> receivedCoordinatesTensor)
{
    QVector<float> identifiedFaceDims;

    /* Find highest confidence bounding box and store its coordinates into a vector */
    float maxConfidence = *std::max_element(receivedConfidenceTensor.constBegin(), receivedConfidenceTensor.constEnd());

    if (maxConfidence > 0) {
        int indexMaxConfidence = receivedConfidenceTensor.indexOf(maxConfidence, 0);

        identifiedFaceDims.push_back(receivedCoordinatesTensor.at(FACE_DETECTION_BOX_INDEX * indexMaxConfidence));
        identifiedFaceDims.push_back(receivedCoordinatesTensor.at((FACE_DETECTION_BOX_INDEX * indexMaxConfidence) + 1));
        identifiedFaceDims.push_back(receivedCoordinatesTensor.at((FACE_DETECTION_BOX_INDEX * indexMaxConfidence) + 2));
        identifiedFaceDims.push_back(receivedCoordinatesTensor.at((FACE_DETECTION_BOX_INDEX * indexMaxConfidence) + 3));
    } else {
        /* Set crop dimensions to Face Landmark input size when a face is not identified */
        identifiedFaceDims.push_back(0);
        identifiedFaceDims.push_back(0);
        identifiedFaceDims.push_back(FACE_LANDMARK_INPUT_SIZE);
        identifiedFaceDims.push_back(FACE_LANDMARK_INPUT_SIZE);
    }

    return identifiedFaceDims;
}

void faceDetection::setFaceCropDims(const QVector<float> &faceCropTensor)
{
    float heightOffset = DETECT_BOUNDING_BOX_INCREASE * frameHeight;
    float widthOffset = DETECT_BOUNDING_BOX_INCREASE * frameWidth;

    /* Increase the dimensions of the bounding box to ensure crop contains entire face */
    faceTopLeftX = faceCropTensor.at(0) - widthOffset;
    faceTopLeftY = faceCropTensor.at(1) - heightOffset;
    faceWidth = faceCropTensor.at(2) + (2 * widthOffset);
    faceHeight = faceCropTensor.at(3) + (2 * heightOffset);

    /* Ensure that cropping coordinates are not outside of the image */
    if (faceTopLeftX < 0)
        faceTopLeftX = 0;

    if (faceTopLeftY < 0)
        faceTopLeftY = 0;

    if ((faceWidth + faceTopLeftX) > frameWidth)
        faceWidth = frameWidth - faceTopLeftX;

    if ((faceHeight + faceTopLeftY) > frameHeight)
        faceHeight = frameHeight - faceTopLeftY;
}

void faceDetection::setIrisCropDims(const QVector<float> &detectedFaceTensor, int receivedStride, int timeElapsed)
{
    QVector <int> pointsEye = { 108, 109, 8, 9, 8, 569, 560, 561 };
    float confidenceValue = detectedFaceTensor.last();
    QVector<float> sortedFaceTensor = sortTensorFaceLandmark(detectedFaceTensor, receivedStride);

    /* Calculate confidence probability by applying sigmoid function */
    float confidenceLevel = 1 / (1 + exp(-confidenceValue));

    timeElaspedFaceLandmark = timeElapsed;

    if (confidenceLevel >= DETECT_THRESHOLD_FACE)
        faceVisible = true;
    else
        faceVisible = false;

    eyeCropCoords.clear();

    foreach (int point, pointsEye) {
        float coordinate = sortedFaceTensor.at(point);
        eyeCropCoords.push_back(coordinate);
    }
}

void faceDetection::setLeftIrisTensor(const QVector<float> &outputIrisLeftTensor, int receivedStride, int receivedTimeElapsed)
{
    timeElaspedIrisLeft = receivedTimeElapsed;

    leftEyeTensor.clear();

    /* Store the left iris x,y coordinates into a vector */
    for (int i = receivedStride; i < outputIrisLeftTensor.size(); i+=3) {
        leftEyeTensor.push_back(outputIrisLeftTensor.at(i));
        leftEyeTensor.push_back(outputIrisLeftTensor.at(i + 1));
    }
}

void faceDetection::detectFaceMode()
{
    detectMode = faceMode;

    uiFD->labelAIModelFilenameFD->setText(TEXT_FACE_MODEL);
    uiFD->labelInferenceTimeFaceDetection->setText(TEXT_INFERENCE_FACE_DETECTION);
    uiFD->labelInferenceTimeFaceLandmark->setText(TEXT_INFERENCE_FACE_LANDMARK);
    uiFD->labelInferenceTimeIrisLandmark->setVisible(false);
    uiFD->labelTotalFpsFace->setText(TEXT_TOTAL_FPS);
    uiFD->graphicsViewPointPlotFace->scene()->clear();
    uiFD->pushButtonDetectFace->setDisabled(true);
    uiFD->pushButtonDetectFace->setStyleSheet(BUTTON_GREYED_OUT);
    uiFD->pushButtonDetectIris->setEnabled(true);
    uiFD->pushButtonDetectIris->setStyleSheet(BUTTON_BLUE);
    uiFD->stackedWidgetFace->setCurrentIndex(STACK_WIDGET_INDEX_FACE_LANDMARK);

    qApp->processEvents();

    if (!continuousMode)
        displayFrame();
}

void faceDetection::detectIrisMode()
{
    detectMode = irisMode;

    uiFD->labelAIModelFilenameFD->setText(TEXT_FACE_MODEL_WITH_IRIS_MODEL);
    uiFD->labelInferenceTimeFaceDetection->setText(TEXT_INFERENCE_FACE_DETECTION);
    uiFD->labelInferenceTimeFaceLandmark->setText(TEXT_INFERENCE_FACE_LANDMARK);
    uiFD->labelInferenceTimeIrisLandmark->setVisible(true);
    uiFD->labelInferenceTimeIrisLandmark->setText(TEXT_INFERENCE_IRIS_LANDMARK);
    uiFD->labelTotalFpsFace->setText(TEXT_TOTAL_FPS);
    uiFD->pushButtonDetectFace->setEnabled(true);
    uiFD->pushButtonDetectFace->setStyleSheet(BUTTON_BLUE);
    uiFD->pushButtonDetectIris->setDisabled(true);
    uiFD->pushButtonDetectIris->setStyleSheet(BUTTON_GREYED_OUT);
    uiFD->stackedWidgetFace->setCurrentIndex(STACK_WIDGET_INDEX_IRIS_LANDMARK);

    qApp->processEvents();

    if (!continuousMode)
        displayFrame();
}

void faceDetection::stopContinuousMode()
{
    continuousMode = false;

    stopVideo();
    setButtonState(true);

    uiFD->labelInferenceTimeFaceDetection->setText(TEXT_INFERENCE_FACE_DETECTION);
    uiFD->labelInferenceTimeFaceLandmark->setText(TEXT_INFERENCE_FACE_LANDMARK);

    if (detectMode == irisMode)
        uiFD->labelInferenceTimeIrisLandmark->setText(TEXT_INFERENCE_IRIS_LANDMARK);

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
                uiFD->labelInferenceTimeFaceDetection->setText(TEXT_INFERENCE_FACE_DETECTION);
                uiFD->labelInferenceTimeFaceLandmark->setText(TEXT_INFERENCE_FACE_LANDMARK);

                if (detectMode == irisMode)
                    uiFD->labelInferenceTimeIrisLandmark->setText(TEXT_INFERENCE_IRIS_LANDMARK);

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

bool faceDetection::getUseIrisMode()
{
    bool useIrisMode;

    if (detectMode == irisMode)
        useIrisMode = true;
    else
        useIrisMode = false;

    return useIrisMode;
}