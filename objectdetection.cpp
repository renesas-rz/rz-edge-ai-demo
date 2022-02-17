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
#include <QFileDialog>

#include "objectdetection.h"
#include "ui_mainwindow.h"

objectDetection::objectDetection(Ui::MainWindow *ui, const QString labelPath)
{
    QFont font;
    QFile labelFile;
    QString fileLine;

    uiOD = ui;
    inputModeOD = cameraModeOD;

    uiOD->actionShopping_Basket->setDisabled(false);
    uiOD->actionObject_Detection->setDisabled(true);
    uiOD->actionLoad_Camera->setDisabled(true);
    uiOD->actionLoad_File->setText(TEXT_LOAD_FILE);
    uiOD->actionLoad_Model->setVisible(true);

    uiOD->labelInference->setText(TEXT_INFERENCE);
    uiOD->labelDemoMode->setText("Mode: Object Detection");
    uiOD->labelTotalFps->setText(TEXT_TOTAL_FPS);

    font.setPointSize(14);
    uiOD->tableWidgetOD->verticalHeader()->setDefaultSectionSize(25);
    uiOD->tableWidgetOD->setHorizontalHeaderLabels({"Object Name", "Count"});
    uiOD->tableWidgetOD->horizontalHeader()->setFont(font);
    uiOD->tableWidgetOD->setEditTriggers(QAbstractItemView::NoEditTriggers);
    uiOD->tableWidgetOD->resizeColumnsToContents();
    double objectNameColumnWidth = uiOD->tableWidgetOD->geometry().width() * 0.8;
    uiOD->tableWidgetOD->setColumnWidth(0, objectNameColumnWidth);
    uiOD->tableWidgetOD->horizontalHeader()->setStretchLastSection(true);
    uiOD->tableWidgetOD->setRowCount(0);

    uiOD->stackedWidgetLeft->setCurrentIndex(1);
    uiOD->stackedWidgetRight->setCurrentIndex(1);

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

    setButtonState(true);
}

void objectDetection::setButtonState(bool enable)
{
    if (enable) {
        buttonState = true;
        uiOD->pushButtonStartStop->setText("Start");
        uiOD->pushButtonStartStop->setStyleSheet(BUTTON_BLUE);
    } else {
        buttonState = false;
        uiOD->pushButtonStartStop->setText("Stop");
        uiOD->pushButtonStartStop->setStyleSheet(BUTTON_RED);
    }

    qApp->processEvents(QEventLoop::WaitForMoreEvents);
}

void objectDetection::triggerInference()
{
    if (inputModeOD == imageModeOD) {
        continuousMode = false;

        stopVideo();
        setButtonState(false);

        emit getFrame();
    } else {
        if (buttonState) {
            continuousMode = true;

            timeTotalFps(true);
            setButtonState(false);
            stopVideo();

            emit getFrame();
        } else {
            continuousMode = false;

            setButtonState(true);

            if (inputModeOD == videoModeOD) {
                stopVideo();
            } else {
                startVideo();
                uiOD->labelInference->setText(TEXT_INFERENCE);
                uiOD->labelTotalFps->setText(TEXT_TOTAL_FPS);
                uiOD->tableWidgetOD->setRowCount(0);
            }
        }
    }
}

void objectDetection::runInference(const QVector<float> &receivedTensor, int receivedTimeElapsed, const cv::Mat &receivedMat)
{
    outputTensor = receivedTensor;

    uiOD->labelInference->setText(TEXT_INFERENCE + QString("%1 ms").arg(receivedTimeElapsed));

    updateObjectList(outputTensor);

    emit sendMatToView(receivedMat);

    if (continuousMode) {
        /* Stop Total FPS timer and display it to GUI, then restart timer before getting the next frame */
        timeTotalFps(false);
        timeTotalFps(true);
        emit getFrame();
    } else {
        setButtonState(true);
    }

    emit getBoxes(outputTensor, labelList);
}

void objectDetection::updateObjectList(const QVector<float> receivedList)
{
    QStringList objectsDetectedList;
    QTableWidgetItem* objectName;
    QTableWidgetItem* objectAmount;

    uiOD->tableWidgetOD->setRowCount(0);

    for (int i = 0; (i + 5) < receivedList.size(); i += 6) {
        objectsDetectedList.append(labelList[int(receivedList[i])]);
    }

    for (int i = 0; i < labelList.size(); i++) {
        int objectTotal = objectsDetectedList.count(labelList.at(i));

        if (objectTotal > 0) {
            objectName = new QTableWidgetItem(labelList.at(i));
            objectAmount = new QTableWidgetItem(QString::number(objectTotal));

            objectName->setTextAlignment(Qt::AlignCenter);
            objectAmount->setTextAlignment(Qt::AlignCenter);

            uiOD->tableWidgetOD->insertRow(uiOD->tableWidgetOD->rowCount());
            uiOD->tableWidgetOD->setItem(uiOD->tableWidgetOD->rowCount()-1, 0, objectName);
            uiOD->tableWidgetOD->setItem(uiOD->tableWidgetOD->rowCount()-1, 1, objectAmount);
        }
    }
    uiOD->tableWidgetOD->sortByColumn(0, Qt::AscendingOrder);
    uiOD->tableWidgetOD->insertRow(uiOD->tableWidgetOD->rowCount());
}

void objectDetection::setCameraMode()
{
    inputModeOD = cameraModeOD;

    uiOD->actionLoad_Camera->setEnabled(false);
    uiOD->actionLoad_File->setText(TEXT_LOAD_FILE);
}

void objectDetection::setImageMode()
{
    inputModeOD = imageModeOD;

    uiOD->actionLoad_Camera->setEnabled(true);
    uiOD->actionLoad_File->setText(TEXT_LOAD_NEW_FILE);
}

void objectDetection::setVideoMode()
{
    inputModeOD = videoModeOD;

    uiOD->actionLoad_Camera->setEnabled(true);
    uiOD->actionLoad_File->setText(TEXT_LOAD_NEW_FILE);
}

void objectDetection::timeTotalFps(bool startingTimer)
{
    if (startingTimer) {
        /* Start the timer to measure Total FPS */
        startTime = std::chrono::high_resolution_clock::now();
    } else {
        /* Stop timer, calculate Total FPS and display to GUI */
        std::chrono::high_resolution_clock::time_point stopTime = std::chrono::high_resolution_clock::now();
        int timeElapsed = int(std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime).count());
        displayTotalFps(timeElapsed);
    }
}

void objectDetection::displayTotalFps(int totalProcessTime)
{
    float totalFps = 1000.0/totalProcessTime;

    uiOD->labelTotalFps->setText(TEXT_TOTAL_FPS + QString::number(double(totalFps), 'f', 1));
}

void objectDetection::stopContinuousMode()
{
    continuousMode = false;

    stopVideo();
    setButtonState(true);

    uiOD->labelInference->setText(TEXT_INFERENCE);
    uiOD->labelTotalFps->setText(TEXT_TOTAL_FPS);
    uiOD->tableWidgetOD->setRowCount(0);

    if (inputModeOD != videoModeOD)
        emit startVideo();
}
