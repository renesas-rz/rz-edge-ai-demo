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
    QFile labelFile;
    QString fileLine;

    uiOD = ui;

    uiOD->actionShopping_Basket->setDisabled(false);
    uiOD->actionObject_Detection->setDisabled(true);
    uiOD->actionLoad_Model->setVisible(true);

    uiOD->labelInference->setText(TEXT_INFERENCE);
    uiOD->labelDemoMode->setText("Mode: Object Detection");
    uiOD->labelTotalFps->setText(TEXT_TOTAL_FPS);

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
    if (buttonState) {
        startTime = std::chrono::high_resolution_clock::now();
        continuousMode = true;
        setButtonState(false);
        stopVideo();

        emit getFrame();
    } else {
        continuousMode = false;
        setButtonState(true);
        startVideo();

        uiOD->labelInference->setText(TEXT_INFERENCE);
        uiOD->labelTotalFps->setText(TEXT_TOTAL_FPS);
    }
}

void objectDetection::runInference(const QVector<float> &receivedTensor, int receivedTimeElapsed, const cv::Mat &receivedMat)
{
    int timeElapsed;
    std::chrono::high_resolution_clock::time_point stopTime;

    stopTime = std::chrono::high_resolution_clock::now();
    timeElapsed = int(std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime).count());

    outputTensor = receivedTensor;

    uiOD->labelInference->setText(TEXT_INFERENCE + QString("%1 ms").arg(receivedTimeElapsed));

    displayTotalFPS(timeElapsed);

    if (continuousMode) {
        emit sendMatToView(receivedMat);
        startTime = std::chrono::high_resolution_clock::now();
        emit getFrame();
    }

    emit getBoxes(outputTensor, labelList);
}

void objectDetection::displayTotalFPS(int totalProcessTime)
{
    float totalFps = 1000.0/totalProcessTime;

    uiOD->labelTotalFps->setText(TEXT_TOTAL_FPS + QString::number(double(totalFps), 'f', 1));
}

void objectDetection::stopContinuousMode()
{
    continuousMode = false;
    setButtonState(true);
    uiOD->labelInference->setText(TEXT_INFERENCE);
    uiOD->labelTotalFps->setText(TEXT_TOTAL_FPS);
    emit startVideo();
}
