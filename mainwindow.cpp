/*****************************************************************************************
 * Copyright (C) 2021 Renesas Electronics Corp.
 * This file is part of the RZG Shopping Basket Demo.
 *
 * The RZG Shopping Basket Demo is free software using the Qt Open Source Model: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * The RZG Shopping Basket Demo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with the RZG Shopping Basket Demo.  If not, see <https://www.gnu.org/licenses/>.
 *****************************************************************************************/

#include <QGraphicsScene>
#include <QGraphicsTextItem>
#include <QFileDialog>
#include <QMessageBox>
#include <QThread>
#include <QEventLoop>
#include <QTimer>
#include <QImageReader>
#include <QElapsedTimer>
#include <QScreen>
#include <QSysInfo>

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "tfliteworker.h"
#include "opencvworker.h"

const QStringList MainWindow::labelList = {"Baked Beans", "Coke", "Diet Coke",
                               "Fusilli Pasta", "Lindt Chocolate",
                               "Mars", "Penne Pasta", "Pringles",
                               "Redbull", "Sweetcorn"};

const std::vector<float> MainWindow::costs = {float(0.85), float(0.82),
                                              float(0.79), float(0.89),
                                              float(1.80), float(0.80),
                                              float(0.89), float(0.85),
                                              float(1.20), float(0.69)};

MainWindow::MainWindow(QWidget *parent, QString cameraLocation, QString modelLocation)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    webcamName = cameraLocation;
    modelPath = modelLocation;
    useArmNNDelegate = true;

    ui->setupUi(this);
    setApplicationSize();
    scene = new QGraphicsScene(this);
    ui->graphicsView->setScene(scene);

    ui->graphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui->graphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    QFont font;
    font.setPointSize(14);
    ui->tableWidget->setHorizontalHeaderLabels({"Item", "Price"});
    ui->tableWidget->horizontalHeader()->setFont(font);
    ui->tableWidget->setEditTriggers(QAbstractItemView::NoEditTriggers);
    ui->tableWidget->resizeColumnsToContents();
    double column1Width = ui->tableWidget->geometry().width() * 0.8;
    ui->tableWidget->setColumnWidth(0, column1Width);
    ui->tableWidget->horizontalHeader()->setStretchLastSection(true);

    ui->labelInference->setText(TEXT_INFERENCE);

    qRegisterMetaType<cv::Mat>();
    opencvThread = new QThread();
    opencvThread->setObjectName("opencvThread");
    opencvThread->start();
    cvWorker = new opencvWorker();
    cvWorker->moveToThread(opencvThread);

    connect(cvWorker, SIGNAL(sendImage(const QImage&)), this, SLOT(showImage(const QImage&)));
    connect(cvWorker, SIGNAL(webcamInit(bool)), this, SLOT(webcamInitStatus(bool)));

    QMetaObject::invokeMethod(cvWorker, "initialiseWebcam", Qt::AutoConnection, Q_ARG(QString, webcamName));

    qRegisterMetaType<QVector<float> >("QVector<float>");

    fpsTimer = new QElapsedTimer();

    QSysInfo systemInfo;

    if (systemInfo.machineHostName() == "hihope-rzg2m") {
        setWindowTitle("Shopping Basket Demo - RZ/G2M");
        boardInfo = G2M_HW_INFO;
    } else if (systemInfo.machineHostName() == "smarc-rzg2l") {
        setWindowTitle("Shopping Basket Demo - RZ/G2L");
        boardInfo = G2L_HW_INFO;
    } else {
        setWindowTitle("Shopping Basket Demo");
        boardInfo = HW_INFO_WARNING;;
    }

    tfliteThread = new QThread();
    tfliteThread->setObjectName("tfliteThread");
    createTfThread();

    QMetaObject::invokeMethod(cvWorker, "readFrame");
}

void MainWindow::createTfThread()
{
    int inferenceThreads = 2;
    tfWorker = new tfliteWorker(modelPath, useArmNNDelegate, inferenceThreads);
    tfliteThread->start();
    tfWorker->moveToThread(tfliteThread);

    /* ArmNN Delegate sets the inference threads to amount of CPU cores
     * of the same type logically group first, which for the RZ/G2L and
     * RZ/G2M is 2 */

    connect(tfWorker, SIGNAL(requestImage()), this, SLOT(receiveRequest()));
    connect(this, SIGNAL(sendImage(const QImage&)), tfWorker, SLOT(receiveImage(const QImage&)));
    connect(tfWorker, SIGNAL(sendOutputTensor(const QVector<float>&, int, const QImage&)),
            this, SLOT(receiveOutputTensor(const QVector<float>&, int, const QImage&)));
    connect(this, SIGNAL(sendNumOfInferenceThreads(int)), tfWorker, SLOT(receiveNumOfInferenceThreads(int)));
}

void MainWindow::setApplicationSize()
{
    QScreen *screen = QGuiApplication::primaryScreen();
    QRect screenGeometry = screen->geometry();
    int height = screenGeometry.height();
    int width = screenGeometry.width();

    if (width == 3840 && height == 2160) { // resolution - 3840x2160
        this->resize(2700, 1350);
        ui->tableWidget->verticalHeader()->setDefaultSectionSize(60);
    } else if (width == 1920 && height == 1080) { // resolution - 1920x1080
        this->resize(1400, 700);
    } else if (width == 1280 && height == 720) { // resolution - 1280x720
        QMainWindow::showMaximized();
        ui->tableWidget->verticalHeader()->setDefaultSectionSize(25);
    } else if (width == 1152 && height == 864) { // resolution - 1152x864
        this->resize(700, 525);
        ui->tableWidget->setFixedWidth(350);
        ui->tableWidget->verticalHeader()->setDefaultSectionSize(25);
    } else if (width == 1024 && height == 768) { // resolution - 1024x768
        this->resize(900, 450);
        ui->tableWidget->setFixedWidth(300);
        ui->tableWidget->verticalHeader()->setDefaultSectionSize(25);
    } else {
        QMainWindow::showMaximized();
    }
}

void MainWindow::receiveRequest()
{
    sendImage(imageToSend);
}

void MainWindow::receiveOutputTensor(const QVector<float>& receivedTensor, int receivedTimeElapsed, const QImage& receivedImage)
{
    QTableWidgetItem* item;
    QTableWidgetItem* price;
    float totalCost = 0;

    outputTensor = receivedTensor;
    ui->tableWidget->setRowCount(0);
    labelListSorted.clear();

    if (webcamDisconnect == true) {
        webcamNotConnected();
    } else {
        for (int i = 0; (i + 5) < receivedTensor.size(); i += 6) {
            totalCost += costs[int(outputTensor[i])];
            labelListSorted.push_back(labelList[int(outputTensor[i])]);
        }

        labelListSorted.sort();

        for (int i = 0; i < labelListSorted.size(); i++) {
            QTableWidgetItem* item = new QTableWidgetItem(labelListSorted.at(i));
            item->setTextAlignment(Qt::AlignCenter);

            ui->tableWidget->insertRow(ui->tableWidget->rowCount());
            ui->tableWidget->setItem(ui->tableWidget->rowCount()-1, 0, item);
            ui->tableWidget->setItem(ui->tableWidget->rowCount()-1, 1,
            price = new QTableWidgetItem("£" + QString::number(
            double(costs[labelList.indexOf(labelListSorted.at(i))]), 'f', 2)));
            price->setTextAlignment(Qt::AlignRight);
        }

        ui->labelInference->setText(TEXT_INFERENCE + QString("%1 ms").arg(receivedTimeElapsed));
        ui->tableWidget->insertRow(ui->tableWidget->rowCount());

        item = new QTableWidgetItem("Total Cost:");
        item->setTextAlignment(Qt::AlignBottom | Qt::AlignRight);
        ui->tableWidget->setItem(ui->tableWidget->rowCount()-1, 0, item);

        item = new QTableWidgetItem("£" + QString::number(double(totalCost), 'f', 2));
        item->setTextAlignment(Qt::AlignBottom | Qt::AlignRight);
        ui->tableWidget->setItem(ui->tableWidget->rowCount()-1, 1, item);

        if (!ui->pushButtonProcessBasket->isEnabled()) {
            image = QPixmap::fromImage(receivedImage);
            scene->clear();
            image.scaled(ui->graphicsView->width(), ui->graphicsView->height(), Qt::KeepAspectRatio);
            scene->addPixmap(image);
            scene->setSceneRect(image.rect());
            QMetaObject::invokeMethod(tfWorker, "process");
        }

        drawBoxes();
    }
}

void MainWindow::showImage(const QImage& imageToShow)
{
    QMetaObject::invokeMethod(cvWorker, "readFrame");

    imageNew = imageToShow;
    setImageSize();

    imageNew = imageNew.scaled(imageWidth, imageHeight);

    imageToSend = imageNew;

    image = QPixmap::fromImage(imageToSend);
    scene->clear();
    scene->addPixmap(image);
    scene->setSceneRect(image.rect());
    drawBoxes();
}

void MainWindow::drawBoxes()
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
}

void MainWindow::on_pushButtonProcessBasket_clicked()
{
    outputTensor.clear();
    ui->tableWidget->setRowCount(0);
    ui->labelInference->setText(TEXT_INFERENCE);

    imageToSend = imageNew;
    image = QPixmap::fromImage(imageToSend);
    scene->clear();
    scene->addPixmap(image);
    scene->setSceneRect(image.rect());

    QMetaObject::invokeMethod(tfWorker, "process");
}

void MainWindow::webcamInitStatus(bool webcamStatus)
{
    webcamDisconnect = false;

    if (!webcamStatus) {
        if (webcamName.isEmpty())
            webcamNotConnected();
    } else {
        cvWorker->checkWebcam();
    }
}

void MainWindow::setImageSize()
{
    imageHeight = ui->graphicsView->height();
    imageWidth = ui->graphicsView->width();
}

void MainWindow::on_actionLicense_triggered()
{
    QMessageBox *msgBox = new QMessageBox(QMessageBox::Information, "License",
                             "Copyright (C) 2021 Renesas Electronics Corp.\n\n"
                             "The RZG Shopping Basket Demo is free software using the Qt Open Source Model: "
                             "you can redistribute it and/or modify "
                             "it under the terms of the GNU General Public License as published by "
                             "the Free Software Foundation, either version 2 of the License, or "
                             "(at your option) any later version.\n\n"
                             "The RZG Shopping Basket Demo is distributed in the hope that it will be useful, "
                             "but WITHOUT ANY WARRANTY; without even the implied warranty of "
                             "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the "
                             "GNU General Public License for more details.\n\n"
                             "You should have received a copy of the GNU General Public License "
                             "along with the RZG Shopping Basket Demo. If not, see https://www.gnu.org/licenses."
                             , QMessageBox::NoButton, this, Qt::Dialog | Qt::FramelessWindowHint);
    msgBox->show();
}

void MainWindow::on_actionHardware_triggered()
{
    QMessageBox *msgBox = new QMessageBox(QMessageBox::Information, "Information", boardInfo,
                                 QMessageBox::NoButton, this, Qt::Dialog | Qt::FramelessWindowHint);
    msgBox->show();
}

void MainWindow::on_actionReset_triggered()
{
    if (webcamName.isEmpty() && QDir("/dev/v4l/by-id").exists()) {
        webcamName = QDir("/dev/v4l/by-id").entryInfoList(QDir::NoDotAndDotDot).at(0).absoluteFilePath();
    }
    QMetaObject::invokeMethod(cvWorker, "initialiseWebcam", Qt::DirectConnection, Q_ARG(QString,webcamName));
}

void MainWindow::webcamNotConnected()
{
    QMessageBox *msgBox = new QMessageBox(QMessageBox::Warning, "Warning", "Webcam not connected", QMessageBox::NoButton, this, Qt::Dialog | Qt::FramelessWindowHint);
    msgBox->show();
    scene->clear();
    ui->graphicsView->setScene(scene);
    ui->tableWidget->clearContents();
    ui->tableWidget->setRowCount(0);
    ui->labelInference->setText(TEXT_INFERENCE);
}

void MainWindow::on_actionDisconnect_triggered()
{
    webcamDisconnect = true;
    QMetaObject::invokeMethod(cvWorker, "disconnectWebcam");

    webcamNotConnected();
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

    /* Remake TfLite data structures with the new ArmNNDelegate settings */
    tfliteThread->quit();

    if(!tfliteThread->wait(800)) // Allow time for the TfLite thread to finish
        qWarning("warning: could not recreate TfLite Thread");

    delete tfWorker;
    createTfThread();
}

void MainWindow::on_actionExit_triggered()
{
    QApplication::quit();
}
