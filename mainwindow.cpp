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
#include "facedetection.h"
#include "objectdetection.h"
#include "opencvworker.h"
#include "poseestimation.h"
#include "videoworker.h"
#include "shoppingbasket.h"

MainWindow::MainWindow(QWidget *parent, QString boardName, QString cameraLocation, QString labelLocation,
                       QString modelLocation, Mode mode, QString pricesFile)
    : QMainWindow(parent),
      ui(new Ui::MainWindow)
{
    board = Unknown;
    inputMode = cameraMode;
    demoMode = mode;
    pricesPath = pricesFile;

    QPixmap splashScreenImage(SPLASH_SCREEN_PATH);

    QSplashScreen *splashScreen = new QSplashScreen(splashScreenImage);
    splashScreen->setAttribute(Qt::WA_DeleteOnClose, true);
    font.setPointSize(18);
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
    scene = new QGraphicsScene(this);
    ui->graphicsView->setScene(scene);
    ui->graphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui->graphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    ui->actionEnable_ArmNN_Delegate->setEnabled(false);
    ui->actionTensorFlow_Lite->setEnabled(true);

#ifndef DUNFELL
    ui->actionTensorflow_Lite_XNNPack_delegate->setEnabled(false);
#endif

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
    else if (demoMode == FD)
        setFaceDetectDelegateType();

    qRegisterMetaType<cv::Mat>();
    cvWorker = new opencvWorker(cameraLocation, board);

    splashScreen->close();

    if (cvWorker->cameraInit() == false) {
        qWarning("Camera not initialising. Quitting.");
        errorPopup(TEXT_CAMERA_INIT_STATUS_ERROR, EXIT_CAMERA_INIT_ERROR);
    } else if (cvWorker->getCameraOpen() == false) {
        qWarning("Camera not opening. Quitting.");
        errorPopup(TEXT_CAMERA_OPENING_ERROR, EXIT_CAMERA_STOPPED_ERROR);
    } else {
        createVideoWorker();
        createTfWorker();

        if (demoMode == SB) {
            /* Set default parameters for other modes */
            labelOD = LABEL_PATH_OD;
            modelOD = MODEL_PATH_OD;

            modelPE = MODEL_PATH_PE_BLAZE_POSE_LITE;

            setupShoppingMode();
        } else if (demoMode == OD) {
            /* Set default parameters for other modes*/
            labelSB = LABEL_PATH_SB;
            modelSB = MODEL_PATH_SB;

            modelPE = MODEL_PATH_PE_BLAZE_POSE_LITE;

            setupObjectDetectMode();
        } else if (demoMode == PE) {
            /* Set default parameters for other modes */
            labelSB = LABEL_PATH_SB;
            modelSB = MODEL_PATH_SB;

            labelOD = LABEL_PATH_OD;
            modelOD = MODEL_PATH_OD;

            setupPoseEstimateMode();
        } else if (demoMode == FD) {
            /* Set default parameters for other modes */
            labelSB = LABEL_PATH_SB;
            modelSB = MODEL_PATH_SB;

            labelOD = LABEL_PATH_OD;
            modelOD = MODEL_PATH_OD;

            modelPE = MODEL_PATH_PE_BLAZE_POSE_LITE;

            setupFaceDetectMode();
        }

        /* Limit camera loop speed if using mipi camera to save on CPU
         * USB camera is alreay limited to 10 FPS */
        if (cvWorker->getUsingMipi())
            vidWorker->setDelayMS(MIPI_VIDEO_DELAY);

        vidWorker->StartVideo();
    }
}

void MainWindow::setupObjectDetectMode()
{
    demoMode = OD;
    tfWorker->setDemoMode(demoMode);

    objectDetectMode = new objectDetection(ui, labelFileList, modelPath, inferenceEngine);

    connect(this, SIGNAL(stopInference()), objectDetectMode, SLOT(stopContinuousMode()), Qt::DirectConnection);
    connect(ui->pushButtonLoadAIModelOD, SIGNAL(pressed()), this, SLOT(loadAIModel()));
    connect(ui->pushButtonStartStop, SIGNAL(pressed()), objectDetectMode, SLOT(triggerInference()));
    connect(objectDetectMode, SIGNAL(getFrame()), this, SLOT(processFrame()), Qt::QueuedConnection);
    connect(objectDetectMode, SIGNAL(getBoxes(QVector<float>,QStringList)), this, SLOT(drawBoxes(QVector<float>,QStringList)));
    connect(objectDetectMode, SIGNAL(sendMatToView(cv::Mat)), this, SLOT(drawMatToView(cv::Mat)));
    connect(objectDetectMode, SIGNAL(startVideo()), vidWorker, SLOT(StartVideo()));
    connect(objectDetectMode, SIGNAL(stopVideo()), vidWorker, SLOT(StopVideo()));
    connect(tfWorker, SIGNAL(sendOutputTensor(const QVector<float>, int, int, const cv::Mat&)),
            objectDetectMode, SLOT(runInference(QVector<float>, int, int, cv::Mat)));
}

void MainWindow::setupShoppingMode()
{
    demoMode = SB;
    tfWorker->setDemoMode(demoMode);

    shoppingBasketMode = new shoppingBasket(ui, labelFileList, pricesPath, modelPath, inferenceEngine);

    connect(ui->pushButtonLoadAIModelSB, SIGNAL(pressed()), this, SLOT(loadAIModel()));
    connect(ui->pushButtonProcessBasket, SIGNAL(pressed()), shoppingBasketMode, SLOT(processBasket()));
    connect(ui->pushButtonNextBasket, SIGNAL(pressed()), shoppingBasketMode, SLOT(nextBasket()));
    connect(shoppingBasketMode, SIGNAL(getFrame()), this, SLOT(processFrame()));
    connect(shoppingBasketMode, SIGNAL(getBoxes(QVector<float>,QStringList)), this, SLOT(drawBoxes(QVector<float>,QStringList)));
    connect(shoppingBasketMode, SIGNAL(getStaticImage()), this, SLOT(getImageFrame()));
    connect(shoppingBasketMode, SIGNAL(sendMatToView(cv::Mat)), this, SLOT(drawMatToView(cv::Mat)));
    connect(shoppingBasketMode, SIGNAL(startVideo()), vidWorker, SLOT(StartVideo()));
    connect(shoppingBasketMode, SIGNAL(stopVideo()), vidWorker, SLOT(StopVideo()));
    connect(tfWorker, SIGNAL(sendOutputTensor(const QVector<float>, int, int, const cv::Mat&)),
            shoppingBasketMode, SLOT(runInference(QVector<float>, int, int, cv::Mat)));
}

void MainWindow::setupPoseEstimateMode()
{
    demoMode = PE;
    tfWorker->setDemoMode(demoMode);

    poseEstimateMode = new poseEstimation(ui, modelPath, inferenceEngine);

    connect(this, SIGNAL(stopInference()), poseEstimateMode, SLOT(stopContinuousMode()), Qt::DirectConnection);
    connect(ui->pushButtonStartStopPose, SIGNAL(pressed()), poseEstimateMode, SLOT(triggerInference()));
    connect(poseEstimateMode, SIGNAL(getFrame()), this, SLOT(processFrame()), Qt::QueuedConnection);
    connect(poseEstimateMode, SIGNAL(sendMatToView(cv::Mat)), this, SLOT(drawMatToView(cv::Mat)));
    connect(poseEstimateMode, SIGNAL(startVideo()), vidWorker, SLOT(StartVideo()));
    connect(poseEstimateMode, SIGNAL(stopVideo()), vidWorker, SLOT(StopVideo()));
    connect(tfWorker, SIGNAL(sendOutputTensor(const QVector<float>, int, int, const cv::Mat&)),
            poseEstimateMode, SLOT(runInference(QVector<float>, int, int, cv::Mat)));
}

void MainWindow::setupFaceDetectMode()
{
    demoMode = FD;
    tfWorker->setDemoMode(demoMode);

    faceDetectMode = new faceDetection(ui, inferenceEngine);

    connect(this, SIGNAL(stopInference()), faceDetectMode, SLOT(stopContinuousMode()), Qt::DirectConnection);
    connect(ui->pushButtonStartStopFace, SIGNAL(pressed()), faceDetectMode, SLOT(triggerInference()));
    connect(faceDetectMode, SIGNAL(getFrame()), this, SLOT(processFrame()), Qt::QueuedConnection);
    connect(faceDetectMode, SIGNAL(sendMatToView(cv::Mat)), this, SLOT(drawMatToView(cv::Mat)));
    connect(faceDetectMode, SIGNAL(startVideo()), vidWorker, SLOT(StartVideo()));
    connect(faceDetectMode, SIGNAL(stopVideo()), vidWorker, SLOT(StopVideo()));
    connect(tfWorker, SIGNAL(sendOutputTensor(const QVector<float>, int, int, const cv::Mat&)),
            faceDetectMode, SLOT(runInference(QVector<float>, int, int, cv::Mat)));
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
    tfWorker = new tfliteWorker(modelPath, delegateType, inferenceThreads);

    connect(tfWorker, SIGNAL(sendInferenceWarning(QString)), this, SLOT(inferenceWarning(QString)));

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
    /*
     * Only enable ArmNN delegate when not using BlazePose/HandPose models on the
     * RZ/G2L and RZ/G2LC platforms as it does not currently support Const
     * Tensors as inputs for Conv2d
     */
    if ((board == G2E || board == G2M) || !modelPath.contains(IDENTIFIER_MOVE_NET)) {
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

void MainWindow::setFaceDetectDelegateType()
{
    /*
     * Do not enable ArmNN delegate when not using Face Detection mode on all
     * platforms as it does not currently support Const Tensors as inputs for Conv2d
     */
    if (delegateType == armNN) {
        delegateType = none;
        ui->actionEnable_ArmNN_Delegate->setEnabled(false);
        ui->actionTensorFlow_Lite->setEnabled(false);
    } else {
        ui->actionEnable_ArmNN_Delegate->setEnabled(false);
    }
}

void MainWindow::ShowVideo()
{
    const cv::Mat* image;

    image = cvWorker->getImage(1);

    if (image == nullptr) {
        qWarning("Camera no longer working. Quitting.");
        errorPopup(TEXT_CAMERA_FAILURE_ERROR, EXIT_CAMERA_STOPPED_ERROR);
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

            pen.setColor(BOX_GREEN);
            pen.setWidth(BOX_WIDTH);

            itemName->setHtml(QString("<div style='background:rgba(0, 0, 0, 100%);font-size:xx-large;'>" +
                                      QString(labelList[int(outputTensor[i + ITEM_OFFSET_SSD])] + " " +
                                      QString::number(double(scorePercentage), 'f', 1) + "%") +
                                      QString("</div>")));
            itemName->setPos(xmin, ymin);
            itemName->setDefaultTextColor(TEXT_GREEN);
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
    msgBox->setFont(font);
    msgBox->show();
}

void MainWindow::on_actionHardware_triggered()
{
    QMessageBox *msgBox = new QMessageBox(QMessageBox::Information, "Information", boardInfo,
                                 QMessageBox::NoButton, this, Qt::Dialog | Qt::FramelessWindowHint);
    msgBox->setFont(font);
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
            emit stopInference();
        } else {
            qWarning("Camera not working. Quitting.");
            errorPopup(TEXT_CAMERA_FAILURE_ERROR, EXIT_CAMERA_STOPPED_ERROR);
        }
    } else {
        tfWorker->receiveImage(*image);
    }
}

void MainWindow::remakeTfWorker()
{
    delete tfWorker;
    createTfWorker();
    disconnectSignals();

    if (demoMode == SB) {
        setupShoppingMode();
    } else if (demoMode == OD) {
        setupObjectDetectMode();
        emit stopInference();
    } else if (demoMode == PE) {
        setupPoseEstimateMode();
        emit stopInference();
    } else if (demoMode == FD) {
        setupFaceDetectMode();
        emit stopInference();
    }

    checkInputMode();
}

void MainWindow::on_actionEnable_ArmNN_Delegate_triggered()
{
    delegateType = armNN;

    ui->actionEnable_ArmNN_Delegate->setEnabled(false);
    ui->actionTensorFlow_Lite->setEnabled(true);
#ifdef DUNFELL
    ui->actionTensorflow_Lite_XNNPack_delegate->setEnabled(true);
#endif

    remakeTfWorker();
}

void MainWindow::on_actionTensorflow_Lite_XNNPack_delegate_triggered()
{
    delegateType = xnnpack;

    if (demoMode == PE)
        setPoseEstimateDelegateType();
    else if (demoMode == FD)
        setFaceDetectDelegateType();
    else
        ui->actionEnable_ArmNN_Delegate->setEnabled(true);

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
        setFaceDetectDelegateType();
    else
        ui->actionEnable_ArmNN_Delegate->setEnabled(true);

    ui->actionTensorFlow_Lite->setEnabled(false);
#ifdef DUNFELL
    ui->actionTensorflow_Lite_XNNPack_delegate->setEnabled(true);
#endif

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
        emit stopInference();
    else
        checkInputMode();
}

void MainWindow::errorPopup(QString errorMessage, int errorCode)
{
    QMessageBox *msgBox = new QMessageBox(QMessageBox::Critical, "Error", errorMessage,
                                 QMessageBox::NoButton, this, Qt::Dialog | Qt::FramelessWindowHint);
    msgBox->setFont(font);
    msgBox->exec();

    exit(errorCode);
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
    /* Store previous demo modes label and model */
    if (demoMode == OD) {
        labelOD = labelPath;
        modelOD = modelPath;
    } else if (demoMode == PE) {
        modelPE = modelPath;

        /*
         * Only enable ArmNN delegate when switching from BlazePose/HandPose models
         * as it does not currently support Const Tensors as inputs for Conv2d
         */
        if ((board == G2E || board == G2M) || !modelPath.contains(IDENTIFIER_MOVE_NET))
            ui->actionEnable_ArmNN_Delegate->setEnabled(true);
    } else if (demoMode == FD) {
        /*
         * If coming from the Face Detection mode, enable ArmNN Delegate which
         * that mode doesn't support
         */
        ui->actionEnable_ArmNN_Delegate->setEnabled(true);
    }

    inputMode = cameraMode;
    modelPath = modelSB;
    labelPath = labelSB;
    labelFileList = readLabelFile(labelSB);

    if (cvWorker->getUsingMipi())
        iterations = 6;
    else
        iterations = 2;

    delete tfWorker;
    createTfWorker();

    disconnectSignals();
    setupShoppingMode();

    cvWorker->useCameraMode();
    vidWorker->StartVideo();
}

void MainWindow::on_actionObject_Detection_triggered()
{
    /* Store previous demo modes label and model */
    if (demoMode == SB) {
        labelSB = labelPath;
        modelSB = modelPath;
    } else if (demoMode == PE) {
        modelPE = modelPath;

        /*
         * Only enable ArmNN delegate when switching from BlazePose/HandPose models
         * as it does not currently support Const Tensors as inputs for Conv2d
         */
        if ((board == G2E || board == G2M) || !modelPath.contains(IDENTIFIER_MOVE_NET))
            ui->actionEnable_ArmNN_Delegate->setEnabled(true);
    } else if (demoMode == FD) {
        /*
         * If coming from the Face Detection mode, enable ArmNN Delegate which
         * that mode doesn't support
         */
        ui->actionEnable_ArmNN_Delegate->setEnabled(true);
    }

    inputMode = cameraMode;
    modelPath = modelOD;
    labelPath = labelOD;
    labelFileList = readLabelFile(labelPath);

    iterations = 1;

    delete tfWorker;
    createTfWorker();

    disconnectSignals();
    setupObjectDetectMode();

    ui->menuInput->menuAction()->setVisible(true);
    cvWorker->useCameraMode();
    vidWorker->StartVideo();
}

void MainWindow::on_actionPose_Estimation_triggered()
{
    /* Store previous demo modes label and model */
    if (demoMode == SB) {
        labelSB = labelPath;
        modelSB = modelPath;
    } else if (demoMode == OD) {
        labelOD = labelPath;
        modelOD = modelPath;
    }

    modelPath = modelPE;
    inputMode = cameraMode;
    iterations = 1;

    setPoseEstimateDelegateType();

    delete tfWorker;
    createTfWorker();

    disconnectSignals();
    setupPoseEstimateMode();

    ui->menuInput->menuAction()->setVisible(true);
    cvWorker->useCameraMode();
    vidWorker->StartVideo();
}

void MainWindow::on_actionFace_Detection_triggered()
{
    /* Store previous demo modes label and model */
    if (demoMode == SB) {
        labelSB = labelPath;
        modelSB = modelPath;
    } else if (demoMode == OD) {
        labelOD = labelPath;
        modelOD = modelPath;
    } else if (demoMode == PE) {
        modelPE = modelPath;
    }

    modelPath = MODEL_PATH_FD_FACE_LANDMARK;
    inputMode = cameraMode;
    iterations = 1;

    setFaceDetectDelegateType();

    delete tfWorker;
    createTfWorker();

    disconnectSignals();
    setupFaceDetectMode();

    ui->menuInput->menuAction()->setVisible(true);
    cvWorker->useCameraMode();
    vidWorker->StartVideo();
}

void MainWindow::loadAIModel()
{
    qeventLoop = new QEventLoop;
    QFileDialog dialog(this);

    connect(this, SIGNAL(modelLoaded()), qeventLoop, SLOT(quit()));

    if (demoMode == OD)
        emit stopInference();

    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setDirectory(MODEL_DIRECTORY);
    dialog.setNameFilter("TFLite Files (*tflite)");
    dialog.setViewMode(QFileDialog::Detail);

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

    emit stopInference();

    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setDirectory(MODEL_DIRECTORY);
    dialog.setNameFilter("TFLite Files (*tflite)");
    dialog.setViewMode(QFileDialog::Detail);

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
    QString mediaFilePath;

    connect(this, SIGNAL(fileLoaded()), qeventLoop, SLOT(quit()));

    if (demoMode != SB)
        emit stopInference();

    vidWorker->StopVideo();

    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setViewMode(QFileDialog::Detail);
    dialog.setDirectory(MEDIA_DIRECTORY);

    mediaFileFilter = IMAGE_FILE_FILTER;

    if (demoMode != SB)
        mediaFileFilter += VIDEO_FILE_FILTER;

    dialog.setNameFilter(mediaFileFilter);

    if (dialog.exec())
        mediaFileName = dialog.selectedFiles().at(0);

    if (mediaFileName.isEmpty()) {
        QMessageBox *msgBox = new QMessageBox(QMessageBox::Warning, "Warning", "Could not identify media file",
                                     QMessageBox::NoButton, this, Qt::Dialog | Qt::FramelessWindowHint);
        msgBox->setFont(font);
        msgBox->exec();
        qeventLoop->exec();
        return;
    }

    mediaFilePath = QDir::current().absoluteFilePath(mediaFileName);
    ui->actionLoad_Camera->setEnabled(true);

    if (dialog.selectedNameFilter().contains("Images")) {
        inputMode = imageMode;
        cvWorker->useImageMode(mediaFilePath);
    } else if (dialog.selectedNameFilter().contains("Videos")) {
        inputMode = videoMode;
        cvWorker->useVideoMode(mediaFilePath);
    }

    checkInputMode();
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

void MainWindow::on_actionLoad_Camera_triggered()
{
    inputMode = cameraMode;

    ui->actionLoad_Camera->setEnabled(false);
    cvWorker->useCameraMode();
    checkInputMode();

    if (demoMode != SB)
        emit stopInference();
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
    } else {
        if (demoMode == OD)
            objectDetectMode->setCameraMode();
        else if (demoMode == SB)
            shoppingBasketMode->setImageMode(false);
        else if (demoMode == PE)
            poseEstimateMode->setCameraMode();
        else if (demoMode == FD)
            faceDetectMode->setCameraMode();
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
