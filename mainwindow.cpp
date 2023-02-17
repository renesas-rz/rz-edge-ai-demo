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

#include <QEventLoop>
#include <QCloseEvent>
#include <QGraphicsScene>
#include <QGraphicsTextItem>
#include <QFileDialog>
#include <QMessageBox>
#include <QSplashScreen>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "audiocommand.h"
#include "facedetection.h"
#include "objectdetection.h"
#include "opencvworker.h"
#include "poseestimation.h"
#include "videoworker.h"
#include "shoppingbasket.h"

#define LABEL_DIRECTORY "/opt/rz-edge-ai-demo/labels/"

#define TEXT_CAMERA_INIT_STATUS_ERROR "Camera Error!\n\n No camera detected, launching in cameraless mode.\n"
#define TEXT_CAMERA_OPENING_ERROR "Camera Error!\n\n Camera not Opening, please check connection and relaunch application.\n"
#define TEXT_CAMERA_FAILURE_ERROR "Camera Error!\n\n Camera has stopped working, please check the connection and relaunch application.\n"
#define TEXT_INFERENCE_ENGINE_TFLITE "TensorFlow Lite"
#define TEXT_INFERENCE_ENGINE_ARMNN_DELEGATE "TensorFlow Lite + ArmNN Delegate"
#define TEXT_INFERENCE_ENGINE_XNNPACK_DELEGATE "TensorFlow Lite + XNNPACK Delegate"

#define IMAGE_FILE_FILTER "Images (*.bmp *.dib *.jpeg *.jpg *.jpe *.png *.pbm *.pgm *.ppm *.sr *.ras *.tiff *.tif)"
#define VIDEO_FILE_FILTER "Videos (*.asf *.avi *.3gp *.mp4 *m4v *.mov *.flv *.mpeg *.mkv *.webm *.mxf *.ogg);;"
#define AUDIO_FILE_FILTER "Audio Files (*.wav)"

#define LABEL_PATH_AC "/opt/rz-edge-ai-demo/labels/audioDemo_labels.txt"

#define MEDIA_DIRECTORY "/opt/rz-edge-ai-demo/media/"
#define MEDIA_DIRECTORY_SB "/opt/rz-edge-ai-demo/media/shopping-basket/"
#define MEDIA_DIRECTORY_FD "/opt/rz-edge-ai-demo/media/face-detection/"
#define MEDIA_DIRECTORY_PE "/opt/rz-edge-ai-demo/media/pose-estimation/"
#define MEDIA_DIRECTORY_OD "/opt/rz-edge-ai-demo/media/object-detection/"
#define MEDIA_DIRECTORY_AC "/opt/rz-edge-ai-demo/media/audio-command/"
#define MODEL_DIRECTORY "/opt/rz-edge-ai-demo/models/"
#define PRICES_DIRECTORY "/opt/rz-edge-ai-demo/prices/"
#define SPLASH_SCREEN_PATH "/opt/rz-edge-ai-demo/logos/rz-splashscreen.png"
#define RENESAS_RZ_LOGO_PATH "/opt/rz-edge-ai-demo/logos/renesas-rz-logo.png"
#define DEFAULT_VIDEO "/opt/rz-edge-ai-demo/media/pose-estimation/exercising_using_battle_ropes.mp4"
#define DEFAULT_WAV_FILE "/opt/rz-edge-ai-demo/media/audio-command/right/right_1.wav"
#define DEFAULT_SBD_IMG "/opt/rz-edge-ai-demo/media/shopping-basket/shopping_items_003.jpg"
#define DEFAULT_FD_VIDEO "/opt/rz-edge-ai-demo/media/face-detection/face_shaking.mp4"
#define MODEL_PATH_FD_FACE_LANDMARK "/opt/rz-edge-ai-demo/models/face_landmark.tflite"

#define CONFIDENCE_OFFSET_SSD 5
#define ITEM_OFFSET_SSD 4

#define G2E_HW_INFO "Hardware Information\n\nBoard: RZ/G2E ek874\nCPUs: 2x Arm Cortex-A53,\nDDR: 2GB"
#define G2L_HW_INFO "Hardware Information\n\nBoard: RZ/G2L smarc-rzg2l-evk\nCPUs: 2x Arm Cortex-A55\nDDR: 2GB"
#define G2LC_HW_INFO "Hardware Information\n\nBoard: RZ/G2LC smarc-rzg2lc-evk\nCPUs: 2x Arm Cortex-A55\nDDR: 1GB"
#define G2M_HW_INFO "Hardware Information\n\nBoard: RZ/G2M hihope-rzg2m\nCPUs: 2x Arm Cortex-A57, 4x Arm Cortex-A53\nDDR: 4GB"
#define HW_INFO_WARNING "Unknown Board!"

#define APP_WIDTH 1275
#define APP_HEIGHT 635
#define BOX_WIDTH 2
#define MIPI_VIDEO_DELAY 50
#define SPLASH_SCREEN_TEXT_SIZE 22

#define MENUBAR_TEXT_SIZE 15
#define METRICS_TABLE_HEADING_SIZE 17
#define METRICS_TABLE_TEXT_SIZE 14
#define BLUE_BUTTON_TEXT_SIZE 17
#define OUTPUT_GRAPH_TEXT_SIZE 16
#define OUTPUT_TABLE_HEADING_SIZE 16
#define OUTPUT_TABLE_TEXT_SIZE 12
#define FILE_DIALOG_TEXT_SIZE 9
#define POPUP_DIALOG_TEXT_SIZE 14

MainWindow::MainWindow(QWidget *parent, QString boardName, QString cameraLocation, QString labelLocation,
                       QString modelLocation, QString videoLocation, Mode mode, QString pricesFile, bool irisOption, bool autoStart)
    : QMainWindow(parent),
      ui(new Ui::MainWindow)
{
    board = Unknown;
    inputMode = cameraMode;
    demoMode = mode;
    pricesPath = pricesFile;
    mediaPath = videoLocation;
    faceDetectIrisMode = irisOption;
    modelPE = MODEL_PATH_PE_BLAZE_POSE_LITE;
    labelOD = LABEL_PATH_OD;
    modelOD = MODEL_PATH_OD;
    labelSB = LABEL_PATH_SB;
    labelAC = LABEL_PATH_AC;
    modelSB = MODEL_PATH_SB;
    scene = new QGraphicsScene(this);
    sceneAC = new QGraphicsScene(this);
    bool mediaExists = QFile::exists(mediaPath);
    audioCommandMode = nullptr;

    QPixmap splashScreenImage(SPLASH_SCREEN_PATH);

    QSplashScreen *splashScreen = new QSplashScreen(splashScreenImage);
    splashScreen->setAttribute(Qt::WA_DeleteOnClose, true);
    font.setPixelSize(SPLASH_SCREEN_TEXT_SIZE);
    splashScreen->setFont(font);
    splashScreen->show();
    splashScreen->showMessage("Loading the \nRZ Edge AI Demo", Qt::AlignCenter, Qt::blue);
    qApp->processEvents();

    labelPath = labelLocation;
    if (labelPath.isEmpty())
        qWarning("Warning: Label file path not provided");

    labelFileList = readLabelFile(labelPath);

    modelPath = modelLocation;
    if (modelPath.isEmpty())
          qWarning("Warning: Model file path not provided");

    delegateType = armNN;

    iterations = 1;

    ui->setupUi(this);
    this->resize(APP_WIDTH, APP_HEIGHT);

    if (demoMode == AC)
        ui->graphicsView->setScene(sceneAC);
    else
        ui->graphicsView->setScene(scene);

    ui->graphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui->graphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    ui->actionEnable_ArmNN_Delegate->setEnabled(false);
    ui->actionTensorFlow_Lite->setEnabled(true);
    ui->actionTensorflow_Lite_XNNPack_delegate->setEnabled(true);

    connect(this, SIGNAL(sendMatToDraw(cv::Mat)), this, SLOT(drawMatToView(cv::Mat)));

    QPixmap rzLogo;
    rzLogo.load(RENESAS_RZ_LOGO_PATH);
    ui->labelRzLogo->setPixmap(rzLogo);

    qRegisterMetaType<QVector<float> >("QVector<float>");

    if (boardName == "hihope-rzg2m") {
        setWindowTitle("RZ Edge AI Demo - RZ/G2M");
        boardInfo = G2M_HW_INFO;
        board = G2M;

        if (cameraLocation.isEmpty()) {
            if(QDir("/dev/v4l/by-id").exists())
                cameraLocation = QDir("/dev/v4l/by-id").entryInfoList(QDir::NoDotAndDotDot).at(0).absoluteFilePath();
            else
                cameraLocation = QString("/dev/video0");
        }
    } else if (boardName == "smarc-rzg2l") {
        setWindowTitle("RZ Edge AI Demo - RZ/G2L");
        boardInfo = G2L_HW_INFO;
        board = G2L;

        if (cameraLocation.isEmpty())
            cameraLocation = QString("/dev/video0");
    } else if (boardName == "smarc-rzg2lc") {
        setWindowTitle("RZ Edge AI Demo - RZ/G2LC");
        boardInfo = G2LC_HW_INFO;
        board = G2LC;

        if (cameraLocation.isEmpty())
            cameraLocation = QString("/dev/video0");

    } else if (boardName == "ek874") {
        setWindowTitle("RZ Edge AI Demo - RZ/G2E");
        boardInfo = G2E_HW_INFO;
        board = G2E;

        if (cameraLocation.isEmpty()) {
            if(QDir("/dev/v4l/by-id").exists())
                cameraLocation = QDir("/dev/v4l/by-id").entryInfoList(QDir::NoDotAndDotDot).at(0).absoluteFilePath();
            else
                cameraLocation = QString("/dev/video0");
        }
    } else {
        setWindowTitle("RZ Edge AI Demo");
        boardInfo = HW_INFO_WARNING;
    }

    if (demoMode == PE)
        setPoseEstimateDelegateType();
    else if (demoMode == FD || demoMode == AC)
        disableArmNNDelegate();

    qRegisterMetaType<size_t>("size_t");
    qRegisterMetaType<cv::Mat>();

    cvWorker = new opencvWorker(cameraLocation, board);
    connect(cvWorker, SIGNAL(resolutionError(QString)), SLOT(errorPopup(QString)), Qt::DirectConnection);

    setGuiPixelSizes();
    splashScreen->close();

    if (cvWorker->cameraInit() == false) {
        qWarning("Camera not initialising. Starting in Cameraless mode.");
        cameraConnect = false;
        errorPopup(TEXT_CAMERA_INIT_STATUS_ERROR);
    } else if (cvWorker->getCameraOpen() == false) {
        qWarning("Camera not opening. Starting in Cameraless mode.");
        cameraConnect = false;
        errorPopup(TEXT_CAMERA_OPENING_ERROR);
    } else {
        cameraConnect = true;
    }

    createVideoWorker();
    createTfWorker();

    if (demoMode == SB)
        setupShoppingMode();
    else if (demoMode == OD)
        setupObjectDetectMode();
    else if (demoMode == PE)
        setupPoseEstimateMode();
    else if (demoMode == FD)
        setupFaceDetectMode();
    else if (demoMode == AC)
        setupAudioCommandMode();

    if (mediaExists) {
        bool video = false;

        if (cameraConnect)
            ui->actionLoad_Periph->setEnabled(true);

        QList<QString> supportedFormats = {".asf", ".avi", ".3gp", ".mp4", ".m4v", ".mov",
                                           ".flv", ".mpeg", ".mkv", ".webm", ".mxf", ".ogg"};

        foreach(QString format, supportedFormats) {
            if (mediaPath.endsWith(format, Qt::CaseInsensitive))
                video = true;
        }

        if (video) {
            inputMode = videoMode;
            mediaExists = cvWorker->useVideoMode(mediaPath);
        } else {
            inputMode = imageMode;
            cvWorker->useImageMode(mediaPath);
        }
    }

    if (cameraConnect && demoMode != AC) {
        /* Limit camera loop speed if using mipi camera to save on CPU
         * USB camera is alreay limited to 10 FPS */
        if (cvWorker->getUsingMipi())
            vidWorker->setDelayMS(MIPI_VIDEO_DELAY);

        if (!mediaExists)
            vidWorker->StartVideo();
    }


    /* If there is no camera or file selected, select a default file */
    if (!mediaExists && !cameraConnect && demoMode != AC) {
        mediaPath = DEFAULT_VIDEO;
        inputMode = videoMode;
        cvWorker->useVideoMode(mediaPath);
    }

    checkInputMode();

    if (demoMode != AC)
        getImageFrame();

    if (autoStart && (mediaExists || cameraConnect || demoMode == AC)) {
        if (demoMode == PE)
            ui->pushButtonStartStopPose->pressed();
        else if (demoMode == OD)
            ui->pushButtonStartStop->pressed();
        else if (demoMode == SB)
            ui->pushButtonProcessBasket->pressed();
        else if (demoMode == FD)
            ui->pushButtonStartStopFace->pressed();
        else if (demoMode == AC)
            ui->pushButtonTalk->pressed();
    }
}

void MainWindow::setGuiPixelSizes()
{
    /* Menu bar */
    font.setPixelSize(MENUBAR_TEXT_SIZE);
    ui->menuBar->setFont(font);

    QList<QAction *> allMenuItems = ui->menuDemoMode->actions();
    allMenuItems.append(ui->menuInput->actions());
    allMenuItems.append(ui->menuInferenceEngine->actions());
    allMenuItems.append(ui->menuAbout->actions());

    foreach (QAction *item, allMenuItems)
        item->setFont(font);

    /* Metrics table */
    font.setPixelSize(METRICS_TABLE_HEADING_SIZE);
    ui->labelInferenceTitleSB->setFont(font);
    ui->labelInferenceTitleFD->setFont(font);
    ui->labelInferenceTitleOD->setFont(font);
    ui->labelInferenceTitlePE->setFont(font);
    ui->labelInferenceTitleAC->setFont(font);
    ui->labelAIModel->setFont(font);
    ui->labelDemoMode->setFont(font);

    ui->labelInferenceTimeOD->setFont(font);
    ui->labelInferenceTimePE->setFont(font);
    ui->labelInferenceTimeSB->setFont(font);
    ui->labelInferenceTimeFD->setFont(font);
    ui->labelInferenceTimeAC->setFont(font);

    font.setPixelSize(METRICS_TABLE_TEXT_SIZE);
    ui->labelInferenceEngineFD->setFont(font);
    ui->labelInferenceEngineOD->setFont(font);
    ui->labelInferenceEnginePE->setFont(font);
    ui->labelInferenceEngineSB->setFont(font);
    ui->labelInferenceEngineAC->setFont(font);

    ui->labelAIModelFilenameFD->setFont(font);
    ui->labelAIModelFilenameOD->setFont(font);
    ui->labelAIModelFilenamePE->setFont(font);
    ui->labelAIModelFilenameSB->setFont(font);
    ui->labelAIModelFilenameAC->setFont(font);

    ui->labelInferenceEngineFD->setFont(font);
    ui->labelInferenceEngineOD->setFont(font);
    ui->labelInferenceEnginePE->setFont(font);
    ui->labelInferenceEngineSB->setFont(font);
    ui->labelInferenceEngineAC->setFont(font);

    /* Face Detection mode */
    ui->labelInferenceTimeFaceDetection->setFont(font);
    ui->labelInferenceTimeFaceLandmark->setFont(font);
    ui->labelInferenceTimeIrisLandmark->setFont(font);

    font.setPixelSize(METRICS_TABLE_TEXT_SIZE);
    ui->labelTotalFpsFace->setFont(font);

    font.setPixelSize(BLUE_BUTTON_TEXT_SIZE);
    ui->pushButtonDetectFace->setFont(font);
    ui->pushButtonDetectIris->setFont(font);
    ui->pushButtonStartStopFace->setFont(font);

    font.setPixelSize(OUTPUT_GRAPH_TEXT_SIZE);
    ui->graphicsViewPointPlotFace->setFont(font);
    ui->labelGraphicalViewTitleFD->setFont(font);

    /* Object Detection mode */
    font.setPixelSize(BLUE_BUTTON_TEXT_SIZE);
    ui->pushButtonLoadAIModelOD->setFont(font);
    ui->pushButtonStartStop->setFont(font);

    font.setPixelSize(OUTPUT_TABLE_HEADING_SIZE);
    ui->tableWidgetOD->setFont(font);

    /* Shopping Basket mode */
    font.setPixelSize(BLUE_BUTTON_TEXT_SIZE);
    ui->pushButtonLoadAIModelSB->setFont(font);
    ui->pushButtonNextBasket->setFont(font);
    ui->pushButtonProcessBasket->setFont(font);

    font.setPixelSize(OUTPUT_TABLE_HEADING_SIZE);
    ui->tableWidget->horizontalHeader()->setFont(font);

    font.setPixelSize(OUTPUT_TABLE_TEXT_SIZE);
    ui->tableWidget->setFont(font);

    font.setPixelSize(METRICS_TABLE_HEADING_SIZE);
    ui->labelTotalItems->setFont(font);
    ui->labelTotalFps->setFont(font);

    /* Pose Estimation mode */
    font.setPixelSize(METRICS_TABLE_HEADING_SIZE);
    ui->labelTotalFpsPose->setFont(font);

    font.setPixelSize(BLUE_BUTTON_TEXT_SIZE);
    ui->pushButtonLoadPoseModel->setFont(font);
    ui->pushButtonStartStopPose->setFont(font);

    font.setPixelSize(OUTPUT_GRAPH_TEXT_SIZE);
    ui->labelGraphicalViewTitle->setFont(font);

    /* Audio Command mode */
    font.setPixelSize(BLUE_BUTTON_TEXT_SIZE);
    ui->pushButtonTalk->setFont(font);

    font.setPixelSize(OUTPUT_TABLE_HEADING_SIZE);
    ui->tableWidgetAC->horizontalHeader()->setFont(font);

    font.setPixelSize(OUTPUT_TABLE_TEXT_SIZE);
    ui->tableWidgetAC->setFont(font);
    ui->commandReaderAC->setFont(font);

    font.setPixelSize(OUTPUT_TABLE_HEADING_SIZE);
    ui->labelCountAC->setFont(font);
    ui->labelHistoryAC->setFont(font);
    ui->commandList->setFont(font);
    ui->noiseLevel->setFont(font);
    ui->micVolume->setFont(font);
}

void MainWindow::setupObjectDetectMode()
{
    demoMode = OD;
    tfWorker->setDemoMode(demoMode);
    ui->graphicsView->setScene(scene);

    checkAudioCommandMode();

    objectDetectMode = new objectDetection(ui, labelFileList, modelPath, inferenceEngine, cameraConnect);

    connect(this, SIGNAL(stopProcessing()), objectDetectMode, SLOT(stopContinuousMode()), Qt::DirectConnection);
    connect(ui->pushButtonLoadAIModelOD, SIGNAL(pressed()), this, SLOT(loadAIModel()));
    connect(ui->pushButtonStartStop, SIGNAL(pressed()), objectDetectMode, SLOT(triggerInference()));
    connect(objectDetectMode, SIGNAL(getFrame()), this, SLOT(processFrame()), Qt::QueuedConnection);
    connect(objectDetectMode, SIGNAL(getBoxes(QVector<float>,QStringList)), this, SLOT(drawBoxes(QVector<float>,QStringList)));
    connect(objectDetectMode, SIGNAL(sendMatToView(cv::Mat)), this, SLOT(drawMatToView(cv::Mat)));
    connect(tfWorker, SIGNAL(sendOutputTensor(const QVector<float>, int, int, const cv::Mat&)),
            objectDetectMode, SLOT(runInference(QVector<float>, int, int, cv::Mat)));

    if (cameraConnect) {
        connect(objectDetectMode, SIGNAL(startVideo()), vidWorker, SLOT(StartVideo()));
        connect(objectDetectMode, SIGNAL(stopVideo()), vidWorker, SLOT(StopVideo()));
    }
}

void MainWindow::setupShoppingMode()
{
    demoMode = SB;
    tfWorker->setDemoMode(demoMode);
    ui->graphicsView->setScene(scene);

    checkAudioCommandMode();

    shoppingBasketMode = new shoppingBasket(ui, labelFileList, pricesPath, modelPath, inferenceEngine, cameraConnect);

    connect(ui->pushButtonLoadAIModelSB, SIGNAL(pressed()), this, SLOT(loadAIModel()));
    connect(ui->pushButtonProcessBasket, SIGNAL(pressed()), shoppingBasketMode, SLOT(processBasket()));
    connect(ui->pushButtonNextBasket, SIGNAL(pressed()), shoppingBasketMode, SLOT(nextBasket()));
    connect(shoppingBasketMode, SIGNAL(getFrame()), this, SLOT(processFrame()));
    connect(shoppingBasketMode, SIGNAL(getBoxes(QVector<float>,QStringList)), this, SLOT(drawBoxes(QVector<float>,QStringList)));
    connect(shoppingBasketMode, SIGNAL(getStaticImage()), this, SLOT(getImageFrame()));
    connect(shoppingBasketMode, SIGNAL(sendMatToView(cv::Mat)), this, SLOT(drawMatToView(cv::Mat)));
    connect(tfWorker, SIGNAL(sendOutputTensor(const QVector<float>, int, int, const cv::Mat&)),
            shoppingBasketMode, SLOT(runInference(QVector<float>, int, int, cv::Mat)));

    if (cameraConnect) {
        connect(shoppingBasketMode, SIGNAL(startVideo()), vidWorker, SLOT(StartVideo()));
        connect(shoppingBasketMode, SIGNAL(stopVideo()), vidWorker, SLOT(StopVideo()));
    }
}

void MainWindow::setupPoseEstimateMode()
{
    demoMode = PE;
    tfWorker->setDemoMode(demoMode);
    ui->graphicsView->setScene(scene);

    checkAudioCommandMode();

    poseEstimateMode = new poseEstimation(ui, modelPath, inferenceEngine, cameraConnect);

    connect(this, SIGNAL(stopProcessing()), poseEstimateMode, SLOT(stopContinuousMode()), Qt::DirectConnection);
    connect(ui->pushButtonStartStopPose, SIGNAL(pressed()), poseEstimateMode, SLOT(triggerInference()));
    connect(poseEstimateMode, SIGNAL(getFrame()), this, SLOT(processFrame()), Qt::QueuedConnection);
    connect(poseEstimateMode, SIGNAL(sendMatToView(cv::Mat)), this, SLOT(drawMatToView(cv::Mat)));
    connect(tfWorker, SIGNAL(sendOutputTensor(const QVector<float>, int, int, const cv::Mat&)),
            poseEstimateMode, SLOT(runInference(QVector<float>, int, int, cv::Mat)));

    if (cameraConnect) {
        connect(poseEstimateMode, SIGNAL(startVideo()), vidWorker, SLOT(StartVideo()));
        connect(poseEstimateMode, SIGNAL(stopVideo()), vidWorker, SLOT(StopVideo()));
    }
}

void MainWindow::setupFaceDetectMode()
{
    DetectMode detectModeToUse;

    demoMode = FD;
    tfWorkerFaceDetection->setDemoMode(demoMode);
    tfWorkerFaceLandmark->setDemoMode(demoMode);
    tfWorkerIrisLandmarkL->setDemoMode(demoMode);
    tfWorkerIrisLandmarkR->setDemoMode(demoMode);
    ui->graphicsView->setScene(scene);

    checkAudioCommandMode();

    if (faceDetectIrisMode)
        detectModeToUse = irisMode;
    else
        detectModeToUse = faceMode;

    faceDetectMode = new faceDetection(ui, inferenceEngine, detectModeToUse, cameraConnect);

    connect(this, SIGNAL(stopProcessing()), faceDetectMode, SLOT(stopContinuousMode()), Qt::DirectConnection);
    connect(ui->pushButtonStartStopFace, SIGNAL(pressed()), faceDetectMode, SLOT(triggerInference()));
    connect(ui->pushButtonDetectFace, SIGNAL(pressed()), faceDetectMode, SLOT(detectFaceMode()));
    connect(ui->pushButtonDetectIris, SIGNAL(pressed()), faceDetectMode, SLOT(detectIrisMode()));
    connect(faceDetectMode, SIGNAL(getFrame()), this, SLOT(processFrame()), Qt::QueuedConnection);
    connect(faceDetectMode, SIGNAL(sendMatToView(cv::Mat)), this, SLOT(drawMatToView(cv::Mat)));
    connect(faceDetectMode, SIGNAL(sendMatForInference(cv::Mat,FaceModel,bool)),
            this, SLOT(runFaceInference(cv::Mat,FaceModel,bool)));
    connect(faceDetectMode, SIGNAL(displayFrame()), this, SLOT(getImageFrame()));
    connect(tfWorkerFaceDetection, SIGNAL(sendOutputTensor(QVector<float>,int,int,cv::Mat)),
            faceDetectMode, SLOT(cropImageFace(QVector<float>,int,int,cv::Mat)));
    connect(tfWorkerIrisLandmarkL, SIGNAL(sendOutputTensorImageless(QVector<float>,int,int)),
            faceDetectMode, SLOT(setLeftIrisTensor(QVector<float>,int,int)));
    connect(tfWorkerIrisLandmarkR, SIGNAL(sendOutputTensorImageless(QVector<float>,int,int)),
            faceDetectMode, SLOT(runInference(QVector<float>,int,int)));

    if (cameraConnect) {
        connect(faceDetectMode, SIGNAL(startVideo()), vidWorker, SLOT(StartVideo()));
        connect(faceDetectMode, SIGNAL(stopVideo()), vidWorker, SLOT(StopVideo()));
    }

    if (detectModeToUse == irisMode) {
        connect(tfWorkerFaceLandmark, SIGNAL(sendOutputTensorImageless(QVector<float>,int,int)),
                faceDetectMode, SLOT(setIrisCropDims(QVector<float>,int,int)));
    } else {
        connect(tfWorkerFaceLandmark, SIGNAL(sendOutputTensorImageless(QVector<float>,int,int)),
                faceDetectMode, SLOT(runInference(QVector<float>, int, int)));
    }
}

void MainWindow::checkAudioCommandMode()
{
    if (audioCommandMode) {
        delete audioCommandMode;
        audioCommandMode = nullptr;
    }
}

void MainWindow::setupAudioCommandMode()
{
    demoMode = AC;
    tfWorker->setDemoMode(demoMode);
    ui->graphicsView->setScene(sceneAC);
    sceneAC->clear();
    labelPath = labelAC;
    labelFileList = readLabelFile(labelAC);

    disableXnnPackDelegate();
    disableArmNNDelegate();

    audioCommandMode = new audioCommand(ui, labelFileList, inferenceEngine);

    connect(audioCommandMode, SIGNAL(requestInference(void*, size_t)), tfWorker, SLOT(processData(void*, size_t)), Qt::QueuedConnection); // this is running on reciever thread, try direct for caller thread
    connect(audioCommandMode, SIGNAL(micWarning(QString)), SLOT(errorPopup(QString)), Qt::DirectConnection);
    connect(tfWorker, SIGNAL(sendOutputTensorBasic(QVector<float>, int)), audioCommandMode, SLOT(interpretInference(QVector<float>, int)), Qt::DirectConnection);
    connect(ui->pushButtonTalk, SIGNAL(pressed()), audioCommandMode, SLOT(toggleAudioInput()));
}

void MainWindow::createVideoWorker()
{
    vidWorker = new videoWorker();

    connect(vidWorker, SIGNAL(showVideo()), this, SLOT(ShowVideo()));
}

void MainWindow::createTfWorker()
{
    /* ArmNN Delegate sets the inference threads to amount of CPU cores
     * of the same type logically group first, which for the RZ/G2L and
     * RZ/G2M is 2 */
    int inferenceThreads = 2;

    if (demoMode == FD) {
        /* Face Detection mode creates two tfliteWorker objects to run the
         * face detection and face landmark models */
        tfWorkerFaceDetection = new tfliteWorker(MODEL_PATH_FD_FACE_DETECTION, delegateType, inferenceThreads);
        tfWorkerFaceLandmark = new tfliteWorker(MODEL_PATH_FD_FACE_LANDMARK, delegateType, inferenceThreads);
        tfWorkerIrisLandmarkL = new tfliteWorker(MODEL_PATH_FD_IRIS_LANDMARK, delegateType, inferenceThreads);
        tfWorkerIrisLandmarkR = new tfliteWorker(MODEL_PATH_FD_IRIS_LANDMARK, delegateType, inferenceThreads);

        connect(tfWorkerFaceDetection, SIGNAL(sendInferenceWarning(QString)), this, SLOT(inferenceWarning(QString)));
        connect(tfWorkerFaceLandmark, SIGNAL(sendInferenceWarning(QString)), this, SLOT(inferenceWarning(QString)));
        connect(tfWorkerIrisLandmarkL, SIGNAL(sendInferenceWarning(QString)), this, SLOT(inferenceWarning(QString)));
        connect(tfWorkerIrisLandmarkR, SIGNAL(sendInferenceWarning(QString)), this, SLOT(inferenceWarning(QString)));
    } else {
        tfWorker = new tfliteWorker(modelPath, delegateType, inferenceThreads);

        connect(tfWorker, SIGNAL(sendInferenceWarning(QString)), this, SLOT(inferenceWarning(QString)));
    }

    if (delegateType == armNN)
        inferenceEngine = TEXT_INFERENCE_ENGINE_ARMNN_DELEGATE;
    else if (delegateType == none)
        inferenceEngine = TEXT_INFERENCE_ENGINE_TFLITE;
    else if (delegateType == xnnpack)
        inferenceEngine = TEXT_INFERENCE_ENGINE_XNNPACK_DELEGATE;
    else
        inferenceEngine = "Unknown inference engine";
}

void MainWindow::setPoseEstimateDelegateType()
{
    /* Only enable ArmNN delegate when not using BlazePose/HandPose models on the
     * RZ/G2L and RZ/G2LC platforms as it does not currently support Const
     * Tensors as inputs for Conv2d */
    if (!modelPath.contains(IDENTIFIER_MOVE_NET)) {
        if (delegateType == armNN) {
            delegateType = none;
            ui->actionEnable_ArmNN_Delegate->setEnabled(false);
            ui->actionTensorFlow_Lite->setEnabled(false);
        } else {
            ui->actionEnable_ArmNN_Delegate->setEnabled(false);
        }
    } else if (modelPath.contains(IDENTIFIER_MOVE_NET) && delegateType != armNN) {
        ui->actionEnable_ArmNN_Delegate->setEnabled(true);
    }
}

void MainWindow::disableArmNNDelegate()
{
    /* Do not enable ArmNN delegate when using modes that require
     * Const Tensors as inputs for Conv2d or dynamic-sized tensors */
    ui->actionEnable_ArmNN_Delegate->setEnabled(false);

    if (delegateType == armNN) {
        delegateType = none;
        ui->actionTensorFlow_Lite->setEnabled(false);
    }
}

void MainWindow::disableXnnPackDelegate()
{
    /* Do not enable XNNPack delegate when using models that require
     * dynamic-sized tensors */
    ui->actionTensorflow_Lite_XNNPack_delegate->setEnabled(false);

    if (delegateType == xnnpack) {
        delegateType = none;
        ui->actionTensorFlow_Lite->setEnabled(false);
    }
}

void MainWindow::ShowVideo()
{
    const cv::Mat* image;

    image = cvWorker->getImage(1);

    if (image == nullptr) {
        qWarning("Camera no longer working.");
        errorPopup(TEXT_CAMERA_FAILURE_ERROR);
    } else {
        emit sendMatToDraw(*image);
    }
}

void MainWindow::drawBoxes(const QVector<float>& outputTensor, QStringList labelList)
{
    for (int i = 0; (i + 5) < outputTensor.size(); i += 6) {
        QPen pen;
        QBrush brush;
        QGraphicsTextItem* itemName = scene->addText(nullptr);
        float ymin = outputTensor[i + 0] * float(scene->height());
        float xmin = outputTensor[i + 1] * float(scene->width());
        float ymax = outputTensor[i + 2] * float(scene->height());
        float xmax = outputTensor[i + 3] * float(scene->width());
        float scorePercentage = outputTensor[i + CONFIDENCE_OFFSET_SSD] * 100;

        pen.setColor(THEME_GREEN);
        pen.setWidth(BOX_WIDTH);

        itemName->setHtml(QString("<div style='background:rgba(0, 0, 0, 100%);font-size:xx-large;'>" +
                                  QString(labelList[int(outputTensor[i + ITEM_OFFSET_SSD])] + " " +
                                  QString::number(double(scorePercentage), 'f', 1) + "%") +
                                  QString("</div>")));
        itemName->setPos(xmin, ymin);
        itemName->setDefaultTextColor(THEME_GREEN);
        itemName->setZValue(1);

        scene->addRect(double(xmin), double(ymin), double(xmax - xmin), double(ymax - ymin), pen, brush);
    }

    if (demoMode == SB)
        ui->labelTotalItems->setText(TEXT_TOTAL_ITEMS + QString("%1").arg(outputTensor.size() / 6));
}

void MainWindow::on_actionLicense_triggered()
{
    QMessageBox *msgBox = new QMessageBox(QMessageBox::Information, "License",
                             "Copyright (C) 2022 Renesas Electronics Corp.\n\n"
                             "The RZ Edge AI Demo is free software using the Qt Open Source Model: "
                             "you can redistribute it and/or modify "
                             "it under the terms of the GNU General Public License as published by "
                             "the Free Software Foundation, either version 2 of the License, or "
                             "(at your option) any later version.\n\n"
                             "The RZ Edge AI Demo is distributed in the hope that it will be useful, "
                             "but WITHOUT ANY WARRANTY; without even the implied warranty of "
                             "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the "
                             "GNU General Public License for more details.\n\n"
                             "You should have received a copy of the GNU General Public License "
                             "along with the RZ Edge AI Demo. If not, see https://www.gnu.org/licenses.",
                             QMessageBox::NoButton, this, Qt::Dialog | Qt::FramelessWindowHint);
    font.setPixelSize(POPUP_DIALOG_TEXT_SIZE);
    msgBox->setFont(font);
    msgBox->show();
}

void MainWindow::on_actionHardware_triggered()
{
    QMessageBox *msgBox = new QMessageBox(QMessageBox::Information, "Information", boardInfo,
                                 QMessageBox::NoButton, this, Qt::Dialog | Qt::FramelessWindowHint);
    msgBox->setFont(font);
    font.setPixelSize(POPUP_DIALOG_TEXT_SIZE);
    msgBox->show();
}

void MainWindow::drawMatToView(const cv::Mat& matInput)
{
    QImage imageToDraw;

    imageToDraw = matToQImage(matInput);

    image = QPixmap::fromImage(imageToDraw);
    scene->clear();

    if ((!cvWorker->getUsingMipi() && inputMode == cameraMode) || (inputMode == imageMode))
        image = image.scaled(GRAPHICS_VIEW_WIDTH, GRAPHICS_VIEW_HEIGHT, Qt::AspectRatioMode::KeepAspectRatio);

    if (demoMode == PE)
        poseEstimateMode->setFrameDims(image.height(), image.width());
    else if (demoMode == FD)
        faceDetectMode->setFrameDims(image.height(), image.width());

    scene->addPixmap(image);
    scene->setSceneRect(image.rect());
}

QImage MainWindow::matToQImage(const cv::Mat& matToConvert)
{
    QImage convertedImage;

    if (matToConvert.empty())
        return QImage(nullptr);

    convertedImage = QImage(matToConvert.data, matToConvert.cols,
                     matToConvert.rows, int(matToConvert.step),
                        QImage::Format_RGB888).copy();

    return convertedImage;
}

void MainWindow::processFrame()
{
    const cv::Mat* image;

    image = cvWorker->getImage(iterations);

    if (image == nullptr) {
        /* Check if video file frame is empty */
        if (inputMode == videoMode) {
            qWarning("Unable to play video file");

            QMessageBox *msgBox = new QMessageBox(QMessageBox::Warning, "Warning", "Could not play video, please check file",
                                         QMessageBox::NoButton, this, Qt::Dialog | Qt::FramelessWindowHint);
            msgBox->setFont(font);
            msgBox->exec();
            emit stopProcessing();
        } else {
            qWarning("Camera not working.");
            errorPopup(TEXT_CAMERA_FAILURE_ERROR);
        }
    } else {
        if (demoMode == FD)
            faceDetectMode->processFace(*image);
        else
            tfWorker->receiveImage(*image);
    }
}

void MainWindow::runFaceInference(const cv::Mat &receivedMat, FaceModel faceModelToUse, bool useIrisModel)
{
    if (faceDetectIrisMode != useIrisModel) {
        faceDetectIrisMode = useIrisModel;

        /* Connect correct signals for Face Detection model modes */
        if (faceDetectIrisMode) {
            disconnect(tfWorkerFaceLandmark, SIGNAL(sendOutputTensorImageless(QVector<float>,int,int)),
                       faceDetectMode, SLOT(runInference(QVector<float>, int, int)));
            connect(tfWorkerFaceLandmark, SIGNAL(sendOutputTensorImageless(QVector<float>,int,int)),
                    faceDetectMode, SLOT(setIrisCropDims(QVector<float>,int,int)));
        } else {
            disconnect(tfWorkerFaceLandmark, SIGNAL(sendOutputTensorImageless(QVector<float>,int,int)),
                       faceDetectMode, SLOT(setIrisCropDims(QVector<float>,int,int)));
            connect(tfWorkerFaceLandmark, SIGNAL(sendOutputTensorImageless(QVector<float>,int,int)),
                    faceDetectMode, SLOT(runInference(QVector<float>, int, int)));
        }
    }

    if (faceModelToUse == faceDetect)
        tfWorkerFaceDetection->receiveImage(receivedMat);
    else if (faceModelToUse == faceLandmark)
        tfWorkerFaceLandmark->receiveImage(receivedMat);
    else if (faceModelToUse == irisLandmarkL)
        tfWorkerIrisLandmarkL->receiveImage(receivedMat);
    else if (faceModelToUse == irisLandmarkR)
        tfWorkerIrisLandmarkR->receiveImage(receivedMat);
}

void MainWindow::deleteTfWorker()
{
    if (demoMode == FD) {
        delete tfWorkerFaceDetection;
        delete tfWorkerFaceLandmark;
        delete tfWorkerIrisLandmarkL;
        delete tfWorkerIrisLandmarkR;
    } else {
        delete tfWorker;
    }
}

void MainWindow::remakeTfWorker()
{
    deleteTfWorker();
    createTfWorker();
    disconnectSignals();

    emit stopProcessing();

    if (demoMode == SB)
        setupShoppingMode();
    else if (demoMode == OD)
        setupObjectDetectMode();
    else if (demoMode == PE)
        setupPoseEstimateMode();
    else if (demoMode == FD)
        setupFaceDetectMode();
    else if (demoMode == AC)
        setupAudioCommandMode();

    checkInputMode();
}

void MainWindow::on_actionEnable_ArmNN_Delegate_triggered()
{
    delegateType = armNN;

    ui->actionEnable_ArmNN_Delegate->setEnabled(false);
    ui->actionTensorFlow_Lite->setEnabled(true);
    ui->actionTensorflow_Lite_XNNPack_delegate->setEnabled(true);

    remakeTfWorker();
}

void MainWindow::on_actionTensorflow_Lite_XNNPack_delegate_triggered()
{
    delegateType = xnnpack;

    if (demoMode == PE)
        setPoseEstimateDelegateType();
    else if (demoMode == FD)
        faceDetectIrisMode = faceDetectMode->getUseIrisMode();
    else
        ui->actionEnable_ArmNN_Delegate->setEnabled(true);

    if(demoMode == FD || demoMode == AC)
        disableArmNNDelegate();

    ui->actionTensorFlow_Lite->setEnabled(true);
    ui->actionTensorflow_Lite_XNNPack_delegate->setEnabled(false);

    remakeTfWorker();
}

void MainWindow::on_actionTensorFlow_Lite_triggered()
{
    delegateType = none;

    if (demoMode == PE)
        setPoseEstimateDelegateType();
    else if (demoMode == FD)
        faceDetectIrisMode = faceDetectMode->getUseIrisMode();
    else
        ui->actionEnable_ArmNN_Delegate->setEnabled(true);

    if (demoMode == FD || demoMode == AC)
        disableArmNNDelegate();

    if (demoMode == AC)
        disableXnnPackDelegate();

    ui->actionTensorFlow_Lite->setEnabled(false);
    ui->actionTensorflow_Lite_XNNPack_delegate->setEnabled(true);

    remakeTfWorker();
}

void MainWindow::inferenceWarning(QString warningMessage)
{
    QMessageBox *msgBox = new QMessageBox(QMessageBox::Critical, "Warning", warningMessage,
                                 QMessageBox::NoButton, this, Qt::Dialog | Qt::FramelessWindowHint);
    msgBox->setFont(font);
    msgBox->exec();

    /* Reset the GUI after showing inference warning */
    if (demoMode == OD || demoMode == PE)
        emit stopProcessing();
    else
        checkInputMode();
}

void MainWindow::errorPopup(QString errorMessage)
{
    QMessageBox *msgBox = new QMessageBox(QMessageBox::Critical, "Error", errorMessage,
                                 QMessageBox::NoButton, this, Qt::Dialog | Qt::FramelessWindowHint);
    msgBox->setFont(font);
    msgBox->exec();
}

void MainWindow::on_actionExit_triggered()
{
    cvWorker->~opencvWorker();
    QApplication::quit();
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    if (demoMode == OD || demoMode == PE)
        cvWorker->useCameraMode();

    event->accept();
}

void MainWindow::on_actionShopping_Basket_triggered()
{
    if (cameraConnect)
        vidWorker->StopVideo();

    ui->menuDemoMode->setEnabled(false);

    /* Store previous demo modes label and model */
    if (demoMode == OD) {
        labelOD = labelPath;
        modelOD = modelPath;
    } else if (demoMode == PE) {
        modelPE = modelPath;

        /* Only enable ArmNN delegate when switching from BlazePose/HandPose models
         * as it does not currently support Const Tensors as inputs for Conv2d */
        if (!modelPath.contains(IDENTIFIER_MOVE_NET))
            ui->actionEnable_ArmNN_Delegate->setEnabled(true);
    } else if (demoMode == FD) {
        faceDetectIrisMode = faceDetectMode->getUseIrisMode();

        /* If coming from the Face Detection mode, enable ArmNN Delegate which
         * that mode doesn't support */
        ui->actionEnable_ArmNN_Delegate->setEnabled(true);
    } else if (demoMode == AC) {
        labelAC = labelPath;

        ui->actionEnable_ArmNN_Delegate->setEnabled(true);
        ui->actionTensorflow_Lite_XNNPack_delegate->setEnabled(true);
    }

    deleteTfWorker();
    disconnectSignals();

    scene->clear();
    demoMode = SB;
    modelPath = modelSB;
    labelPath = labelSB;
    labelFileList = readLabelFile(labelSB);

    if (cameraConnect)
        inputMode = cameraMode;
    else
        inputMode = imageMode;

    if (cvWorker->getUsingMipi())
        iterations = 6;
    else
        iterations = 2;

    createTfWorker();
    setupShoppingMode();

    if (cameraConnect) {
        cvWorker->useCameraMode();
        vidWorker->StartVideo();
    } else if (inputMode == imageMode) {
        cvWorker->useImageMode(DEFAULT_SBD_IMG);
    }

    ui->menuDemoMode->setEnabled(true);
}

void MainWindow::startDefaultMode()
{
    ui->menuInput->menuAction()->setVisible(true);

    if (inputMode == cameraMode) {
        cvWorker->useCameraMode();
        vidWorker->StartVideo();
    } else if (inputMode == videoMode) {
        cvWorker->useVideoMode(mediaPath);
    }
}

void MainWindow::on_actionObject_Detection_triggered()
{
    ui->menuDemoMode->setEnabled(false);

    if (cameraConnect)
        vidWorker->StopVideo();

    /* Store previous demo modes label and model */
    if (demoMode == SB) {
        labelSB = labelPath;
        modelSB = modelPath;
    } else if (demoMode == PE) {
        modelPE = modelPath;

        /* Only enable ArmNN delegate when switching from BlazePose/HandPose models
         * as it does not currently support Const Tensors as inputs for Conv2d */
        if (!modelPath.contains(IDENTIFIER_MOVE_NET))
            ui->actionEnable_ArmNN_Delegate->setEnabled(true);
    } else if (demoMode == FD) {
        faceDetectIrisMode = faceDetectMode->getUseIrisMode();

        /* If coming from the Face Detection mode, enable ArmNN Delegate which
         * that mode doesn't support */
        ui->actionEnable_ArmNN_Delegate->setEnabled(true);
    } else if (demoMode == AC) {
        labelAC = labelPath;

        ui->actionEnable_ArmNN_Delegate->setEnabled(true);
        ui->actionTensorflow_Lite_XNNPack_delegate->setEnabled(true);
    }

    deleteTfWorker();
    disconnectSignals();

    scene->clear();
    demoMode = OD;
    modelPath = modelOD;
    labelPath = labelOD;
    mediaPath = DEFAULT_VIDEO;
    labelFileList = readLabelFile(labelPath);

    iterations = 1;

    if (cameraConnect)
        inputMode = cameraMode;
    else
        inputMode = videoMode;

    createTfWorker();
    setupObjectDetectMode();
    startDefaultMode();

    ui->menuDemoMode->setEnabled(true);

    emit stopProcessing();
}

void MainWindow::on_actionPose_Estimation_triggered()
{
    ui->menuDemoMode->setEnabled(false);

    if (cameraConnect)
        vidWorker->StopVideo();

    /* Store previous demo modes label and model */
    if (demoMode == SB) {
        labelSB = labelPath;
        modelSB = modelPath;
    } else if (demoMode == OD) {
        labelOD = labelPath;
        modelOD = modelPath;
    } else if (demoMode == FD) {
        faceDetectIrisMode = faceDetectMode->getUseIrisMode();
    } else if (demoMode == AC) {
        labelAC = labelPath;

        ui->actionTensorflow_Lite_XNNPack_delegate->setEnabled(true);
    }

    deleteTfWorker();
    disconnectSignals();

    scene->clear();
    demoMode = PE;
    modelPath = modelPE;
    mediaPath = DEFAULT_VIDEO;
    iterations = 1;

    if (cameraConnect)
        inputMode = cameraMode;
    else
        inputMode = videoMode;

    setPoseEstimateDelegateType();
    createTfWorker();
    setupPoseEstimateMode();
    startDefaultMode();

    ui->menuDemoMode->setEnabled(true);

    emit stopProcessing();
}

void MainWindow::on_actionFace_Detection_triggered()
{
    ui->menuDemoMode->setEnabled(false);

    if (cameraConnect)
        vidWorker->StopVideo();

    /* Store previous demo modes label and model */
    if (demoMode == SB) {
        labelSB = labelPath;
        modelSB = modelPath;
    } else if (demoMode == OD) {
        labelOD = labelPath;
        modelOD = modelPath;
    } else if (demoMode == PE) {
        modelPE = modelPath;
    } else if (demoMode == AC) {
        labelAC = labelPath;
    }

    deleteTfWorker();
    disconnectSignals();

    scene->clear();
    demoMode = FD;
    modelPath = MODEL_PATH_FD_FACE_LANDMARK;
    mediaPath = DEFAULT_FD_VIDEO;
    iterations = 1;

    if (cameraConnect)
        inputMode = cameraMode;
    else
        inputMode = videoMode;

    disableArmNNDelegate();
    createTfWorker();
    setupFaceDetectMode();
    startDefaultMode();

    ui->menuDemoMode->setEnabled(true);

    emit stopProcessing();
}

void MainWindow::on_actionAudio_Command_triggered()
{
    ui->menuDemoMode->setEnabled(false);
    emit stopProcessing();

    if (cameraConnect)
        vidWorker->StopVideo();

    /* Store previous demo modes label and model */
    if (demoMode == SB) {
        labelSB = labelPath;
        modelSB = modelPath;
    } else if (demoMode == OD) {
        labelOD = labelPath;
        modelOD = modelPath;
    } else if (demoMode == FD) {
        faceDetectIrisMode = faceDetectMode->getUseIrisMode();
    } else if (demoMode == PE) {
        modelPE = modelPath;
    }

    deleteTfWorker();
    disconnectSignals();

    demoMode = AC;
    modelPath = MODEL_PATH_AC;
    mediaPath = DEFAULT_WAV_FILE;
    labelPath = labelAC;
    inputMode = audioFileMode;

    disableArmNNDelegate();
    disableXnnPackDelegate();
    createTfWorker();
    setupAudioCommandMode();
    startDefaultMode();

    ui->menuDemoMode->setEnabled(true);
}

void MainWindow::loadAIModel()
{
    qeventLoop = new QEventLoop;
    QFileDialog dialog(this);

    connect(this, SIGNAL(modelLoaded()), qeventLoop, SLOT(quit()));

    if (demoMode == OD)
        emit stopProcessing();

    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setDirectory(MODEL_DIRECTORY);
    dialog.setNameFilter("TFLite Files (*tflite)");
    dialog.setViewMode(QFileDialog::List);
    font.setPixelSize(FILE_DIALOG_TEXT_SIZE);
    dialog.setFont(font);

    modelPath.clear();

    if (dialog.exec())
        modelPath = dialog.selectedFiles().at(0);

    /* Set default model if model path is empty */
    if (modelPath.isEmpty()) {
        if (demoMode == OD)
            modelPath = MODEL_PATH_OD;
        else if (demoMode == SB)
            modelPath = MODEL_PATH_SB;
    }

    dialog.setDirectory(LABEL_DIRECTORY);
    dialog.setNameFilter("Text Files (*txt)");

    labelPath.clear();

    if (dialog.exec())
        labelPath = dialog.selectedFiles().at(0);

    /* Set default label if label path is empty */
    if (labelPath.isEmpty()) {
        if (demoMode == OD)
            labelPath = LABEL_PATH_OD;
        else if (demoMode == SB)
            labelPath = LABEL_PATH_SB;
    }

    labelFileList = readLabelFile(labelPath);

    delete tfWorker;
    createTfWorker();
    disconnectSignals();

    if (demoMode == OD) {
        setupObjectDetectMode();
    } else if (demoMode == SB) {
        /* Prices file selection */
        dialog.setDirectory(PRICES_DIRECTORY);
        dialog.setNameFilter("Text Files (*txt)");

        pricesPath.clear();

        if (dialog.exec())
            pricesPath = dialog.selectedFiles().at(0);

        if (pricesPath.isEmpty())
            pricesPath = PRICES_PATH_DEFAULT;

        setupShoppingMode();
    }

    dialog.close();
    checkInputMode();
    modelLoaded();
    qeventLoop->exec();
}

void MainWindow::on_pushButtonLoadPoseModel_clicked()
{
    QStringList supportedModels = { MODEL_PATH_PE_MOVE_NET_L, MODEL_PATH_PE_MOVE_NET_T, MODEL_PATH_PE_BLAZE_POSE_FULL,
                                    MODEL_PATH_PE_BLAZE_POSE_HEAVY, MODEL_PATH_PE_BLAZE_POSE_LITE,
                                    MODEL_PATH_PE_HAND_POSE_FULL, MODEL_PATH_PE_HAND_POSE_LITE };

    qeventLoop = new QEventLoop;
    QFileDialog dialog(this);

    connect(this, SIGNAL(modelLoaded()), qeventLoop, SLOT(quit()));

    emit stopProcessing();

    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setDirectory(MODEL_DIRECTORY);
    dialog.setNameFilter("TFLite Files (*tflite)");
    dialog.setViewMode(QFileDialog::List);
    font.setPixelSize(FILE_DIALOG_TEXT_SIZE);
    dialog.setFont(font);

    modelPath.clear();

    if (dialog.exec())
        modelPath = dialog.selectedFiles().at(0);

    /* Check if pose model selected is supported */
    if (!(supportedModels.contains(modelPath))) {
        if (modelPath.isEmpty())
            qWarning("Warning: Model file path not provided");
        else
            qWarning("Warning: Unsupported pose model selected");

        modelPath = MODEL_PATH_PE_BLAZE_POSE_LITE;
    }

    setPoseEstimateDelegateType();

    delete tfWorker;
    createTfWorker();
    disconnectSignals();
    setupPoseEstimateMode();

    dialog.close();
    checkInputMode();
    modelLoaded();
    qeventLoop->exec();
}

void MainWindow::on_actionLoad_File_triggered()
{
    qeventLoop = new QEventLoop;
    QFileDialog dialog(this);
    QString mediaFileFilter;
    QString mediaFileName;
    QString mediaDir;

    connect(this, SIGNAL(fileLoaded()), qeventLoop, SLOT(quit()));

    if (demoMode != SB)
        emit stopProcessing();

    if (cameraConnect)
        vidWorker->StopVideo();

    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setViewMode(QFileDialog::List);
    font.setPixelSize(FILE_DIALOG_TEXT_SIZE);
    dialog.setFont(font);

    switch (demoMode)
    {
       case SB: mediaDir = MEDIA_DIRECTORY_SB;
         break;
       case OD: mediaDir = MEDIA_DIRECTORY_OD;
         break;
       case PE: mediaDir = MEDIA_DIRECTORY_PE;
         break;
       case FD: mediaDir = MEDIA_DIRECTORY_FD;
         break;
       case AC: mediaDir = MEDIA_DIRECTORY_AC;
         break;
       default: mediaDir = MEDIA_DIRECTORY;
    }
    dialog.setDirectory(mediaDir);

    if (demoMode == AC) {
        mediaFileFilter += AUDIO_FILE_FILTER;
    } else {
        if (demoMode != SB)
            mediaFileFilter += VIDEO_FILE_FILTER;

        mediaFileFilter += IMAGE_FILE_FILTER;
    }

    dialog.setNameFilter(mediaFileFilter);

    if (dialog.exec())
        mediaFileName = dialog.selectedFiles().at(0);

    if (mediaFileName.isEmpty()) {
        QMessageBox *msgBox = new QMessageBox(QMessageBox::Warning, "Warning", "Could not identify media file",
                                     QMessageBox::NoButton, this, Qt::Dialog | Qt::FramelessWindowHint);
        font.setPixelSize(POPUP_DIALOG_TEXT_SIZE);
        msgBox->setFont(font);
        msgBox->exec();
        qeventLoop->exec();
        return;
    }

    mediaPath = QDir::current().absoluteFilePath(mediaFileName);

    if (cameraConnect && demoMode != AC)
        ui->actionLoad_Periph->setEnabled(true);

    if (dialog.selectedNameFilter().contains("Images")) {
        inputMode = imageMode;
        cvWorker->useImageMode(mediaPath);
    } else if (dialog.selectedNameFilter().contains("Videos")) {
        if (!cvWorker->useVideoMode(mediaPath)) {
            vidWorker->StopVideo();

            /* If there is an attached camera return to that, otherwise
             * give the user the opportunity to select another file */
            if (cameraConnect)
                on_actionLoad_Periph_triggered();
            else
                on_actionLoad_File_triggered();

            return;
        }

        inputMode = videoMode;
    } else if (dialog.selectedNameFilter().contains("Audio Files")) {
        inputMode = audioFileMode;

        audioCommandMode->readAudioFile(mediaPath);
    }

    checkInputMode();

    if (demoMode != AC)
        getImageFrame();

    emit fileLoaded();
    ui->labelTotalFps->setText(TEXT_TOTAL_FPS);
    dialog.close();
    qeventLoop->exec();
}

void MainWindow::getImageFrame()
{
    emit sendMatToDraw(*cvWorker->getImage(1));
}

QStringList MainWindow::readLabelFile(QString labelPath)
{
    QFile labelFile;
    QString fileLine;
    QStringList labelList;

    labelFile.setFileName(labelPath);
    if (!labelFile.open(QIODevice::ReadOnly | QIODevice::Text))
        qFatal("%s could not be opened.", labelPath.toStdString().c_str());

    while (!labelFile.atEnd()) {
        fileLine = labelFile.readLine();
        fileLine.remove(QRegularExpression("^\\s*\\d*\\s*"));
        fileLine.remove(QRegularExpression("\n"));
        labelList.append(fileLine);
    }

    labelFile.close();

    return labelList;
}

void MainWindow::on_actionLoad_Periph_triggered()
{
    if (demoMode == AC)
        inputMode = micMode;
    else
        inputMode = cameraMode;

    ui->actionLoad_Periph->setEnabled(false);

    if (demoMode != AC)
        cvWorker->useCameraMode();

    checkInputMode();

    if (demoMode != SB)
        emit stopProcessing();
}

void MainWindow::checkInputMode()
{
    /* Check to see if a media file is currently loaded */
    if (inputMode == videoMode) {
        vidWorker->StopVideo();

        if (demoMode == OD)
            objectDetectMode->setVideoMode();
        else if (demoMode == PE)
            poseEstimateMode->setVideoMode();
        else if (demoMode == FD)
            faceDetectMode->setVideoMode();
    } else if (inputMode == imageMode) {
        if (demoMode == OD)
            objectDetectMode->setImageMode();
        else if (demoMode == SB)
            shoppingBasketMode->setImageMode(true);
        else if (demoMode == PE)
            poseEstimateMode->setImageMode();
        else if (demoMode == FD)
            faceDetectMode->setImageMode();
    } else if (inputMode == cameraMode) {
        if (demoMode == OD)
            objectDetectMode->setCameraMode();
        else if (demoMode == SB)
            shoppingBasketMode->setImageMode(false);
        else if (demoMode == PE)
            poseEstimateMode->setCameraMode();
        else if (demoMode == FD)
            faceDetectMode->setCameraMode();
    } else if (inputMode == micMode) {
        audioCommandMode->setMicMode();
    }
}

void MainWindow::disconnectSignals()
{
    if (demoMode == SB) {
        shoppingBasketMode->disconnect();

        delete shoppingBasketMode;
    } else if (demoMode == OD) {
        objectDetectMode->disconnect();

        delete objectDetectMode;
    } else if (demoMode == PE) {
        poseEstimateMode->disconnect();

        delete poseEstimateMode;
    } else if (demoMode == FD) {
        faceDetectMode->disconnect();

        delete faceDetectMode;
    }
}
