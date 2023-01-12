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

#define TEXT_FACE_MODEL "face_detection_short_range.tflite\nface_landmark.tflite"
#define TEXT_FACE_MODEL_WITH_IRIS_MODEL "\nface_detection_short_range.tflite\nface_landmark.tflite\niris_landmark.tflite"
#define TEXT_INFERENCE_FACE_DETECTION "Face Detection: "
#define TEXT_INFERENCE_FACE_LANDMARK "Face Landmark: "
#define TEXT_INFERENCE_IRIS_LANDMARK "Iris Landmark: "

namespace Ui { class MainWindow; }

enum DetectMode { faceMode, irisMode };

class faceDetection : public QObject
{
    Q_OBJECT

public:
    faceDetection(Ui::MainWindow *ui, QString inferenceEngine, DetectMode detectModeToUse, bool cameraConnect);
    void processFace(const cv::Mat &matToProcess);
    void setCameraMode();
    void setImageMode();
    void setVideoMode();
    void setFrameDims(int height, int width);
    bool getUseIrisMode();

public slots:
    void runInference(const QVector<float>& receivedTensor, int receivedStride, int receivedTimeElapsed);
    void cropImageFace(const QVector<float> &faceDetectOutputTensor, int receivedStride, int receivedTimeElapsed, const cv::Mat &receivedMat);
    void setFaceCropDims(const QVector<float>& faceCropTensor);
    void setIrisCropDims(const QVector<float>& detectedFaceTensor, int receivedStride, int timeElapsed);
    void setLeftIrisTensor(const QVector<float>& outputIrisLeftTensor, int receivedStride, int receivedTimeElapsed);
    void detectFaceMode();
    void detectIrisMode();
    void stopContinuousMode();
    void triggerInference();

signals:
    void getFrame();
    void sendMatForInference(const cv::Mat &receivedMat, FaceModel faceModelToUse, bool useFaceDetection);
    void sendMatToView(const cv::Mat&receivedMat);
    void startVideo();
    void stopVideo();
    void displayFrame();

private:
    void setButtonState(bool enable);
    QVector<float> sortTensorFaceLandmark(const QVector<float> receivedTensor, int receivedStride);
    QVector<float> sortTensorIrisLandmark(const QVector<float> receivedTensor, int receivedStride);
    void drawPointsFaceLandmark(const QVector<float>& outputTensor, bool updateGraphicalView);
    void drawPointsIrisLandmark(const QVector<float>& outputTensor, bool drawLeftEye);
    void connectLandmarks(int landmark1, int landmark2, bool drawGraphicalViewLandmarks);
    void processIris(const cv::Mat &resizedInputMat, const cv::Mat &croppedFaceMat, bool detectIris);
    QVector<QPair<float, float>> generateAnchorCoords(int inputHeight, int inputWidth);
    QVector<float> sortBoundingBoxes(const QVector<float> receivedConfidenceTensor, const QVector<float> receivedCoordinatesTensor);
    void updateFrameWithoutInference();

    Ui::MainWindow *uiFD;
    Input inputModeFD;
    FaceModel faceModel;
    DetectMode detectMode;
    edgeUtils *utilFD;
    QVector<float> outputTensor;
    QVector<float> xCoordinate;
    QVector<float> yCoordinate;
    QVector<float> eyeCropCoords;
    QVector<float> leftEyeTensor;
    QList<QVector<int>> *faceParts;
    cv::Mat resizedMat;
    bool continuousMode;
    bool buttonState;
    bool faceVisible;
    bool camConnect;
    int frameHeight;
    int frameWidth;
    int timeElaspedFaceDetection;
    int timeElaspedFaceLandmark;
    int timeElaspedIrisLeft;
    float faceHeight;
    float faceWidth;
    float faceTopLeftX;
    float faceTopLeftY;
};

#endif // FACEDETECTION_H
