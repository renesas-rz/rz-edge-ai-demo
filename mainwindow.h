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
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/videoio.hpp>

#include "tfliteworker.h"
#include "edge-utils.h"

#define LABEL_PATH_OD "/opt/rz-edge-ai-demo/labels/mobilenet_ssd_v2_coco_quant_postprocess_labels.txt"
#define LABEL_PATH_SB "/opt/rz-edge-ai-demo/labels/shoppingBasketDemo_labels.txt"

#define MODEL_PATH_OD "/opt/rz-edge-ai-demo/models/mobilenet_ssd_v2_coco_quant_postprocess.tflite"
#define MODEL_PATH_SB "/opt/rz-edge-ai-demo/models/shoppingBasketDemo.tflite"
#define PRICES_PATH_DEFAULT "/opt/rz-edge-ai-demo/prices/shoppingBasketDemo_prices_gbp.txt"
#define MODEL_PATH_PE_MOVE_NET_T "/opt/rz-edge-ai-demo/models/lite-model_movenet_singlepose_thunder_tflite_int8_4.tflite"
#define MODEL_PATH_PE_MOVE_NET_L "/opt/rz-edge-ai-demo/models/lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite"
#define MODEL_PATH_PE_BLAZE_POSE_LITE "/opt/rz-edge-ai-demo/models/pose_landmark_lite.tflite"
#define MODEL_PATH_PE_BLAZE_POSE_HEAVY "/opt/rz-edge-ai-demo/models/pose_landmark_heavy.tflite"
#define MODEL_PATH_PE_BLAZE_POSE_FULL "/opt/rz-edge-ai-demo/models/pose_landmark_full.tflite"
#define MODEL_PATH_PE_HAND_POSE_LITE "/opt/rz-edge-ai-demo/models/hand_landmark_lite.tflite"
#define MODEL_PATH_PE_HAND_POSE_FULL "/opt/rz-edge-ai-demo/models/hand_landmark_full.tflite"
#define MODEL_PATH_AC "/opt/rz-edge-ai-demo/models/browserfft-speech-renesas.tflite"
#define MODEL_PATH_FD_IRIS_LANDMARK "/opt/rz-edge-ai-demo/models/iris_landmark.tflite"

class QGraphicsScene;
class QGraphicsView;
class faceDetection;
class objectDetection;
class audioCommand;
class opencvWorker;
class poseEstimation;
class shoppingBasket;
class tfliteWorker;
class QElapsedTimer;
class QEventLoop;
class videoWorker;

namespace Ui { class MainWindow; } //Needed for mainwindow.ui

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent, QString boardName, QString cameraLocation, QString labelLocation,
               QString modelLocation, QString videoLocation, Mode mode, QString pricesFile, bool irisOption, bool autoStart);

public slots:
    void ShowVideo();
    void processFrame();

signals:
    void fileLoaded();
    void modelLoaded();
    void stopProcessing();
    void sendMatToDraw(const cv::Mat& matToSend);

private slots:
    void closeEvent(QCloseEvent *event);
    void drawBoxes(const QVector<float>& outputTensor, QStringList labelList);
    void drawMatToView(const cv::Mat& matInput);
    void getImageFrame();
    void loadAIModel();
    void runFaceInference(const cv::Mat& receivedMat, FaceModel faceModelToUse, bool useIrisModel);
    void inferenceWarning(QString warningMessage);
    void on_actionLicense_triggered();
    void on_actionEnable_ArmNN_Delegate_triggered();
    void on_actionTensorFlow_Lite_triggered();
    void on_actionTensorflow_Lite_XNNPack_delegate_triggered();
    void on_actionShopping_Basket_triggered();
    void on_actionObject_Detection_triggered();
    void on_actionPose_Estimation_triggered();
    void on_actionFace_Detection_triggered();
    void on_actionAudio_Command_triggered();
    void on_actionHardware_triggered();
    void on_actionExit_triggered();
    void on_actionLoad_Periph_triggered();
    void on_actionLoad_File_triggered();
    void on_pushButtonLoadPoseModel_clicked();
    void errorPopup(QString errorMessage);

private:
    void createTfWorker();
    QImage matToQImage(const cv::Mat& matToConvert);
    void createVideoWorker();
    void deleteTfWorker();
    void remakeTfWorker();
    void setupFaceDetectMode();
    void setupObjectDetectMode();
    void setupPoseEstimateMode();
    void setupShoppingMode();
    void setupAudioCommandMode();
    void disconnectSignals();
    void checkInputMode();
    void setPoseEstimateDelegateType();
    void disableArmNNDelegate();
    void disableXnnPackDelegate();
    void startDefaultMode();
    void setGuiPixelSizes();
    QStringList readLabelFile(QString labelPath);

    Ui::MainWindow *ui;
    unsigned int iterations;
    Delegate delegateType;
    QFont font;
    QPixmap image;
    QGraphicsScene *scene;
    QGraphicsScene *sceneAC;
    QGraphicsView *graphicsView;
    opencvWorker *cvWorker;
    shoppingBasket *shoppingBasketMode;
    objectDetection *objectDetectMode;
    poseEstimation *poseEstimateMode;
    faceDetection *faceDetectMode;
    audioCommand *audioCommandMode;
    tfliteWorker *tfWorker;
    tfliteWorker *tfWorkerFaceDetection;
    tfliteWorker *tfWorkerFaceLandmark;
    tfliteWorker *tfWorkerIrisLandmarkL;
    tfliteWorker *tfWorkerIrisLandmarkR;
    QEventLoop *qeventLoop;
    QString boardInfo;
    QString modelPath;
    QString pricesPath;
    QString mediaPath;
    QString modelOD;
    QString modelPE;
    QString modelSB;
    QString labelPath;
    QString labelOD;
    QString labelSB;
    QString labelAC;
    QString inferenceEngine;
    QStringList labelFileList;
    bool faceDetectIrisMode;
    bool cameraConnect;
    videoWorker *vidWorker;
    Board board;
    Input inputMode;
    Mode demoMode;
};

#endif // MAINWINDOW_H
