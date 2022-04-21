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
#include <QSysInfo>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "objectdetection.h"
#include "opencvworker.h"
#include "videoworker.h"
#include "shoppingbasket.h"

MainWindow::MainWindow(QWidget *parent, QString cameraLocation, QString labelLocation,
                       QString modelLocation, Mode mode, QString pricesFile)
    : QMainWindow(parent),
      ui(new Ui::MainWindow)
{
    Board board = Unknown;
    inputMode = cameraMode;
    demoMode = mode;
    pricesPath = pricesFile;

    QPixmap splashScreenImage(SPLASH_SCREEN_DIRECTORY);

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
    rzLogo.load(RENESAS_RZ_LOGO_DIRECTORY);
    ui->labelRzLogo->setPixmap(rzLogo);

    qRegisterMetaType<QVector<float> >("QVector<float>");

    QSysInfo systemInfo;

    if (systemInfo.machineHostName() == "hihope-rzg2m") {
        setWindowTitle("RZ Edge AI Demo - RZ/G2M");
        boardInfo = G2M_HW_INFO;
        board = G2M;

        if (cameraLocation.isEmpty()) {
            if(QDir("/dev/v4l/by-id").exists())
                cameraLocation = QDir("/dev/v4l/by-id").entryInfoList(QDir::NoDotAndDotDot).at(0).absoluteFilePath();
            else
                cameraLocation = QString("/dev/video0");
        }

    } else if (systemInfo.machineHostName() == "smarc-rzg2l") {
        setWindowTitle("RZ Edge AI Demo - RZ/G2L");
        boardInfo = G2L_HW_INFO;
        board = G2L;

        if (cameraLocation.isEmpty())
            cameraLocation = QString("/dev/video0");

    } else if (systemInfo.machineHostName() == "smarc-rzg2lc") {
        setWindowTitle("RZ Edge AI Demo - RZ/G2LC");
        boardInfo = G2LC_HW_INFO;
        board = G2LC;

        if (cameraLocation.isEmpty())
            cameraLocation = QString("/dev/video0");

    } else if (systemInfo.machineHostName() == "ek874") {
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
            /* Set default object detection parameters */
            labelOD = LABEL_DIRECTORY_OD;
            modelOD = MODEL_DIRECTORY_OD;

            setupShoppingMode();
        } else if (demoMode == OD) {
            /* Set default shopping basket parameters */
            labelSB = LABEL_DIRECTORY_SB;
            modelSB = MODEL_DIRECTORY_SB;

            setupObjectDetectMode();
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
    updateAIModelLabel();

    objectDetectMode = new objectDetection(ui, labelFileList);

    connect(this, SIGNAL(stopInference()), objectDetectMode, SLOT(stopContinuousMode()), Qt::DirectConnection);
    connect(ui->pushButtonLoadAIModelOD, SIGNAL(pressed()), this, SLOT(loadAIModel()));
    connect(ui->pushButtonStartStop, SIGNAL(pressed()), objectDetectMode, SLOT(triggerInference()));
    connect(objectDetectMode, SIGNAL(getFrame()), this, SLOT(processFrame()), Qt::QueuedConnection);
    connect(objectDetectMode, SIGNAL(getBoxes(QVector<float>,QStringList)), this, SLOT(drawBoxes(QVector<float>,QStringList)));
    connect(objectDetectMode, SIGNAL(sendMatToView(cv::Mat)), this, SLOT(drawMatToView(cv::Mat)));
    connect(objectDetectMode, SIGNAL(startVideo()), vidWorker, SLOT(StartVideo()));
    connect(objectDetectMode, SIGNAL(stopVideo()), vidWorker, SLOT(StopVideo()));
    connect(tfWorker, SIGNAL(sendOutputTensor(const QVector<float>, int, const cv::Mat&)),
            objectDetectMode, SLOT(runInference(QVector<float>,int,cv::Mat)));
}

void MainWindow::setupShoppingMode()
{
    demoMode = SB;
    updateAIModelLabel();

    shoppingBasketMode = new shoppingBasket(ui, labelFileList, pricesPath);

    connect(ui->pushButtonLoadAIModelSB, SIGNAL(pressed()), this, SLOT(loadAIModel()));
    connect(ui->pushButtonProcessBasket, SIGNAL(pressed()), shoppingBasketMode, SLOT(processBasket()));
    connect(ui->pushButtonNextBasket, SIGNAL(pressed()), shoppingBasketMode, SLOT(nextBasket()));
    connect(shoppingBasketMode, SIGNAL(getFrame()), this, SLOT(processFrame()));
    connect(shoppingBasketMode, SIGNAL(getBoxes(QVector<float>,QStringList)), this, SLOT(drawBoxes(QVector<float>,QStringList)));
    connect(shoppingBasketMode, SIGNAL(getStaticImage()), this, SLOT(getImageFrame()));
    connect(shoppingBasketMode, SIGNAL(sendMatToView(cv::Mat)), this, SLOT(drawMatToView(cv::Mat)));
    connect(shoppingBasketMode, SIGNAL(startVideo()), vidWorker, SLOT(StartVideo()));
    connect(shoppingBasketMode, SIGNAL(stopVideo()), vidWorker, SLOT(StopVideo()));
    connect(tfWorker, SIGNAL(sendOutputTensor(const QVector<float>, int, const cv::Mat&)),
            shoppingBasketMode, SLOT(runInference(QVector<float>,int,cv::Mat)));
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

            pen.setColor(BOX_COLOUR);
            pen.setWidth(BOX_WIDTH);

            itemName->setHtml(QString("<div style='background:rgba(0, 0, 0, 100%);font-size:xx-large;'>" +
                                      QString(labelList[int(outputTensor[i + ITEM_OFFSET_SSD])] + " " +
                                      QString::number(double(scorePercentage), 'f', 1) + "%") +
                                      QString("</div>")));
            itemName->setPos(xmin, ymin);
            itemName->setDefaultTextColor(TEXT_COLOUR);
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
        image = image.scaled(800, 600, Qt::AspectRatioMode::KeepAspectRatio);

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

    ui->labelDelegate->setText("TensorFlow Lite + ArmNN delegate");

    remakeTfWorker();
}

void MainWindow::on_actionTensorflow_Lite_XNNPack_delegate_triggered()
{
    delegateType = xnnpack;

    ui->actionEnable_ArmNN_Delegate->setEnabled(true);
    ui->actionTensorFlow_Lite->setEnabled(true);
    ui->actionTensorflow_Lite_XNNPack_delegate->setEnabled(false);
    ui->labelDelegate->setText("TensorFlow Lite + XNNPack Delegate");

    remakeTfWorker();
}

void MainWindow::on_actionTensorFlow_Lite_triggered()
{
    delegateType = none;

    ui->actionEnable_ArmNN_Delegate->setEnabled(true);
    ui->actionTensorFlow_Lite->setEnabled(false);
#ifdef DUNFELL
    ui->actionTensorflow_Lite_XNNPack_delegate->setEnabled(true);
#endif

    ui->labelDelegate->setText("TensorFlow Lite");

    remakeTfWorker();
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
    if (demoMode == OD)
        cvWorker->useCameraMode();

    event->accept();
}

void MainWindow::on_actionShopping_Basket_triggered()
{
    /* Store objection detection label and model being used */
    labelOD = labelPath;
    modelOD = modelPath;

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
    /* Store shopping basket label and model being used */
    labelSB = labelPath;
    modelSB = modelPath;

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

void MainWindow::loadAIModel()
{
    qeventLoop = new QEventLoop;
    QFileDialog dialog(this);

    connect(this, SIGNAL(modelLoaded()), qeventLoop, SLOT(quit()));

    if (demoMode == OD)
        emit stopInference();

    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setDirectory(MODEL_DIRECTORY_PATH);
    dialog.setNameFilter("TFLite Files (*tflite)");
    dialog.setViewMode(QFileDialog::Detail);

    modelPath.clear();

    if (dialog.exec())
        modelPath = dialog.selectedFiles().at(0);

    /* Set default model if model path is empty */
    if (modelPath.isEmpty()) {
        if (demoMode == OD)
            modelPath = MODEL_DIRECTORY_OD;
        else if (demoMode == SB)
            modelPath = MODEL_DIRECTORY_SB;
    }

    dialog.setDirectory(LABEL_DIRECTORY_PATH);
    dialog.setNameFilter("Text Files (*txt)");

    labelPath.clear();

    if (dialog.exec())
        labelPath = dialog.selectedFiles().at(0);

    /* Set default label if label path is empty */
    if (labelPath.isEmpty()) {
        if (demoMode == OD)
            labelPath = LABEL_DIRECTORY_OD;
        else if (demoMode == SB)
            labelPath = LABEL_DIRECTORY_SB;
    }

    labelFileList = readLabelFile(labelPath);

    delete tfWorker;
    createTfWorker();
    disconnectSignals();

    if (demoMode == OD) {
        setupObjectDetectMode();
    } else if (demoMode == SB) {
        /* Prices file selection */
        dialog.setDirectory(PRICES_DIRECTORY_PATH);
        dialog.setNameFilter("Text Files (*txt)");

        pricesPath.clear();

        if (dialog.exec())
            pricesPath = dialog.selectedFiles().at(0);

        if (pricesPath.isEmpty())
            pricesPath = PRICES_DIRECTORY_DEFAULT;

        setupShoppingMode();
    }

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

    if (demoMode == OD)
        emit stopInference();

    vidWorker->StopVideo();

    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setViewMode(QFileDialog::Detail);
    dialog.setDirectory(MEDIA_DIRECTORY_PATH);

    mediaFileFilter = IMAGE_FILE_FILTER;

    if (demoMode == OD)
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

        if (demoMode == OD)
            objectDetectMode->setImageMode();
        else if (demoMode == SB)
            shoppingBasketMode->setImageMode(true);
    } else if (dialog.selectedNameFilter().contains("Videos")) {
        inputMode = videoMode;
        cvWorker->useVideoMode(mediaFilePath);
        objectDetectMode->setVideoMode();
    }

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

void MainWindow::updateAIModelLabel()
{
    QString modelName = modelPath.section('/', -1);

    ui->labelAIModelFilename->setText(modelName);
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

    if (demoMode == OD)
        emit stopInference();

    checkInputMode();
}

void MainWindow::checkInputMode()
{
    /* Check to see if a media file is currently loaded */
    if (inputMode == videoMode) {
        vidWorker->StopVideo();
        objectDetectMode->setVideoMode();
    } else if (inputMode == imageMode) {
        if (demoMode == OD)
            objectDetectMode->setImageMode();
        else if (demoMode == SB)
            shoppingBasketMode->setImageMode(true);
    } else {
        if (demoMode == OD)
            objectDetectMode->setCameraMode();
        else if (demoMode == SB)
            shoppingBasketMode->setImageMode(false);
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
    }
}
