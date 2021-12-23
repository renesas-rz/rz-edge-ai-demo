/*****************************************************************************************
 * Copyright (C) 2021 Renesas Electronics Corp.
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
#include "tfliteworker.h"
#include "opencvworker.h"
#include "videoworker.h"
#include "shoppingbasket.h"

MainWindow::MainWindow(QWidget *parent, QString cameraLocation, QString modelLocation)
    : QMainWindow(parent),
      ui(new Ui::MainWindow)
{
    Board board = Unknown;

    QPixmap splashScreenImage("/opt/rz-edge-ai-demo/logos/rz-splashscreen.png");

    QSplashScreen *splashScreen = new QSplashScreen(splashScreenImage);
    splashScreen->setAttribute(Qt::WA_DeleteOnClose, true);
    splashScreen->show();
    splashScreen->showMessage("Loading the \nRZ Edge AI Demo", Qt::AlignCenter, Qt::blue);
    qApp->processEvents();
    font.setPointSize(18);
    splashScreen->setFont(font);

    modelPath = modelLocation;
    useArmNNDelegate = true;

    ui->setupUi(this);
    this->resize(APP_WIDTH, APP_HEIGHT);
    scene = new QGraphicsScene(this);
    ui->graphicsView->setScene(scene);
    ui->graphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui->graphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    ui->labelDemoMode->setText("Mode: Shopping Basket");
    ui->menuMode->menuAction()->setVisible(false);

    QPixmap rzLogo;
    rzLogo.load("/opt/rz-edge-ai-demo/logos/renesas-rz-logo.png");
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
        board = G2L;

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
        setupShoppingMode();
        createVideoWorker();
        createTfWorker();

        /* Limit camera loop speed if using mipi camera to save on CPU
         * USB camera is alreay limited to 10 FPS */
        if (cvWorker->getUsingMipi())
            vidWorker->setDelayMS(MIPI_VIDEO_DELAY);

        /* If a Mipi camera is not in use then hide the menu that
         * is only supported for the OV5645 */
        if (!cvWorker->getUsingMipi())
            ui->menuCam_Settings->menuAction()->setVisible(false);

        vidWorker->StartVideo();
    }
}

void MainWindow::setupShoppingMode()
{
    shoppingBasketMode = new shoppingBasket(ui);

    connect(ui->pushButtonProcessBasket, SIGNAL(pressed()), shoppingBasketMode, SLOT(processBasket()));
    connect(ui->pushButtonNextBasket, SIGNAL(pressed()), shoppingBasketMode, SLOT(nextBasket()));
    connect(shoppingBasketMode, SIGNAL(getFrame()), this, SLOT(processFrame()));
    connect(shoppingBasketMode, SIGNAL(getBoxes(QVector<float>,QStringList)), this, SLOT(drawBoxes(QVector<float>,QStringList)));
    connect(shoppingBasketMode, SIGNAL(sendMatToView(cv::Mat)), this, SLOT(sendMatToDraw(cv::Mat)));
}

void MainWindow::createVideoWorker()
{
    vidWorker = new videoWorker();

    connect(vidWorker, SIGNAL(showVideo()), this, SLOT(ShowVideo()));
    connect(shoppingBasketMode, SIGNAL(startVideo()), vidWorker, SLOT(StartVideo()));
    connect(shoppingBasketMode, SIGNAL(stopVideo()), vidWorker, SLOT(StopVideo()));
}

void MainWindow::createTfWorker()
{
    /* ArmNN Delegate sets the inference threads to amount of CPU cores
     * of the same type logically group first, which for the RZ/G2L and
     * RZ/G2M is 2 */
    int inferenceThreads = 2;
    tfWorker = new tfliteWorker(modelPath, useArmNNDelegate, inferenceThreads);

    connect(tfWorker, SIGNAL(sendOutputTensor(const QVector<float>&, int, const cv::Mat&)),
            shoppingBasketMode, SLOT(runInference(QVector<float>,int,cv::Mat)));
}

void MainWindow::ShowVideo()
{
    const cv::Mat* image;

    image = cvWorker->getImage(1);

    if (image == nullptr) {
        qWarning("Camera no longer working. Quitting.");
        errorPopup(TEXT_CAMERA_FAILURE_ERROR, EXIT_CAMERA_STOPPED_ERROR);
    } else {
        drawMatToView(*image);
    }
}

void MainWindow::drawBoxes(const QVector<float>& outputTensor, QStringList labelList)
{
    for (int i = 0; (i + 5) < outputTensor.size(); i += 6) {
            QPen pen;
            QBrush brush;
            QGraphicsTextItem* itemName = scene->addText(nullptr);
            float ymin = outputTensor[i + 2] * float(scene->height());
            float xmin = outputTensor[i + 3] * float(scene->width());
            float ymax = outputTensor[i + 4] * float(scene->height());
            float xmax = outputTensor[i + 5] * float(scene->width());
            float scorePercentage = outputTensor[i + 1] * 100;

            pen.setColor(BOX_COLOUR);
            pen.setWidth(BOX_WIDTH);

            itemName->setHtml(QString("<div style='background:rgba(0, 0, 0, 100%);font-size:xx-large;'>" +
                                      QString(labelList[int(outputTensor[i])] + " " +
                                      QString::number(double(scorePercentage), 'f', 1) + "%") +
                                      QString("</div>")));
            itemName->setPos(xmin, ymin);
            itemName->setDefaultTextColor(TEXT_COLOUR);
            itemName->setZValue(1);

            scene->addRect(double(xmin), double(ymin), double(xmax - xmin), double(ymax - ymin), pen, brush);
        }

        ui->labelTotalItems->setText(TEXT_TOTAL_ITEMS + QString("%1").arg(outputTensor.size() / 6));
}

void MainWindow::on_actionLicense_triggered()
{
    QMessageBox *msgBox = new QMessageBox(QMessageBox::Information, "License",
                             "Copyright (C) 2021 Renesas Electronics Corp.\n\n"
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

void MainWindow::sendMatToDraw(const cv::Mat &matToSend)
{
    drawMatToView(matToSend);
}

void MainWindow::drawMatToView(const cv::Mat& matInput)
{
    QImage imageToDraw;

    imageToDraw = matToQImage(matInput);

    image = QPixmap::fromImage(imageToDraw);
    scene->clear();

    if (!cvWorker->getUsingMipi())
        image = image.scaled(800, 600);

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
    unsigned int iterations;

    if (cvWorker->getUsingMipi())
        iterations = 6;
    else
        iterations = 2;

    image = cvWorker->getImage(iterations);

    if (image == nullptr) {
        qWarning("Camera not working. Quitting.");
        errorPopup(TEXT_CAMERA_FAILURE_ERROR, EXIT_CAMERA_STOPPED_ERROR);
    } else {
        tfWorker->receiveImage(*image);
    }
}

void MainWindow::on_actionEnable_ArmNN_Delegate_triggered()
{
    /* Update the GUI text */
    if(useArmNNDelegate) {
        ui->actionEnable_ArmNN_Delegate->setText("Enable ArmNN Delegate");
        ui->labelDelegate->setText("TensorFlow Lite");
    } else {
        ui->actionEnable_ArmNN_Delegate->setText("Disable ArmNN Delegate");
        ui->labelDelegate->setText("TensorFlow Lite + ArmNN delegate");
    }

    /* Toggle delegate state */
    useArmNNDelegate = !useArmNNDelegate;

    delete tfWorker;
    createTfWorker();
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
    QApplication::quit();
}

void MainWindow::on_actionAuto_White_Balance_triggered()
{
    if (ui->actionAuto_White_Balance->text().contains("Disable"))
        ui->actionAuto_White_Balance->setText("Enable Auto White Balance");
    else
        ui->actionAuto_White_Balance->setText("Disable Auto White Balance");

    cvWorker->toggleWhitebalanceAuto();
}

void MainWindow::on_actionAuto_Exposure_triggered()
{
    if (ui->actionAuto_Exposure->text().contains("Disable"))
        ui->actionAuto_Exposure->setText("Enable Auto Exposure");
    else
        ui->actionAuto_Exposure->setText("Disable Auto Exposure");

    cvWorker->toggleExpose();
}

void MainWindow::on_actionAuto_Gain_triggered()
{
    if (ui->actionAuto_Gain->text().contains("Disable"))
        ui->actionAuto_Gain->setText("Enable Auto Gain");
    else
        ui->actionAuto_Gain->setText("Disable Auto Gain");

    cvWorker->toggleGain();
}

void MainWindow::on_actionShopping_Basket_triggered()
{
    setupShoppingMode();

    ui->labelDemoMode->setText("Mode: Shopping Basket");
    ui->stackedWidgetLeft->setCurrentIndex(0);
    ui->stackedWidgetRight->setCurrentIndex(0);
    ui->actionShopping_Basket->setDisabled(true);
}
