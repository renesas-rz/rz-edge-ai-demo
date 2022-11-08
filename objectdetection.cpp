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

#include "objectdetection.h"
#include "ui_mainwindow.h"

#define ITEM_OFFSET 4
#define BOX_POINTS 4

#define DETECT_THRESHOLD 0.5

objectDetection::objectDetection(Ui::MainWindow *ui, QStringList labelFileList, QString modelPath,
                                 QString inferenceEngine, bool cameraConnect)
{
    QFont font;
    QString modelName;

    uiOD = ui;
    inputModeOD = cameraMode;
    labelList = labelFileList;
    camConnect = cameraConnect;

    utilOD = new edgeUtils();

    modelName = modelPath.section('/', -1);

    uiOD->actionShopping_Basket->setDisabled(false);
    uiOD->actionObject_Detection->setDisabled(true);
    uiOD->actionPose_Estimation->setDisabled(false);
    uiOD->actionFace_Detection->setDisabled(false);
    uiOD->actionLoad_Camera->setDisabled(true);
    uiOD->actionLoad_File->setText(TEXT_LOAD_FILE);

    uiOD->labelAIModelFilenameOD->setText(modelName);
    uiOD->labelInferenceEngineOD->setText(inferenceEngine);
    uiOD->labelInferenceTimeOD->setText(TEXT_INFERENCE);
    uiOD->labelDemoMode->setText("Mode: Object Detection");
    uiOD->labelTotalFps->setText(TEXT_TOTAL_FPS);

    uiOD->tableWidgetOD->verticalHeader()->setDefaultSectionSize(25);
    uiOD->tableWidgetOD->setHorizontalHeaderLabels({"Object Name", "Count"});
    uiOD->tableWidgetOD->setEditTriggers(QAbstractItemView::NoEditTriggers);
    uiOD->tableWidgetOD->resizeColumnsToContents();
    double objectNameColumnWidth = uiOD->tableWidgetOD->geometry().width() * 0.8;
    uiOD->tableWidgetOD->setColumnWidth(0, objectNameColumnWidth);
    uiOD->tableWidgetOD->horizontalHeader()->setStretchLastSection(true);
    uiOD->tableWidgetOD->setRowCount(0);

    uiOD->stackedWidgetLeft->setCurrentIndex(STACK_WIDGET_INDEX_OD);
    uiOD->stackedWidgetRight->setCurrentIndex(STACK_WIDGET_INDEX_OD);

    setButtonState(true);
}

void objectDetection::setButtonState(bool enable)
{
    buttonState = enable;

    if (buttonState) {
        uiOD->pushButtonStartStop->setText("Start\nInference");
        uiOD->pushButtonStartStop->setStyleSheet(BUTTON_BLUE);
    } else {
        uiOD->pushButtonStartStop->setText("Stop\nInference");
        uiOD->pushButtonStartStop->setStyleSheet(BUTTON_RED);
    }

    qApp->processEvents(QEventLoop::WaitForMoreEvents);
}

void objectDetection::triggerInference()
{
    if (inputModeOD == imageMode) {
        continuousMode = false;

        stopVideo();
        setButtonState(false);

        emit getFrame();
    } else {
        if (buttonState) {
            continuousMode = true;

            utilOD->timeTotalFps(true);
            setButtonState(false);
            stopVideo();

            emit getFrame();
        } else {
            continuousMode = false;

            setButtonState(true);

            if (inputModeOD == videoMode) {
                stopVideo();
            } else {
                startVideo();
                uiOD->labelInferenceTimeOD->setText(TEXT_INFERENCE);
                uiOD->labelTotalFps->setText(TEXT_TOTAL_FPS);
                uiOD->tableWidgetOD->setRowCount(0);
            }
        }
    }
}

QVector<float> objectDetection::sortTensor(QVector<float> &receivedTensor, int receivedStride)
{
    QVector<float> sortedTensor = QVector<float>();

    /* The final output tensor of the model is unused in this demo mode */
    receivedTensor.removeLast();

    for(int i = receivedStride; i > 0; i--) {
        float confidenceLevel = receivedTensor.at(receivedTensor.size() - i);

        /* Only include the item if the confidence level is at threshold */
        if (confidenceLevel > DETECT_THRESHOLD && confidenceLevel <= float(1.0)) {
            /* Box points */
            for(int j = 0; j < BOX_POINTS; j++)
                sortedTensor.push_back(receivedTensor.at((receivedStride - i) * BOX_POINTS + j));

            /* Item ID */
            sortedTensor.push_back(receivedTensor.at(receivedTensor.size() - ((receivedStride * 2)) + (receivedStride - i)));

            /* Confidence level */
            sortedTensor.push_back(confidenceLevel);
        }
    }

    return sortedTensor;
}

void objectDetection::runInference(QVector<float> receivedTensor, int receivedStride, int receivedTimeElapsed, const cv::Mat &receivedMat)
{
    float totalFps;

    outputTensor = sortTensor(receivedTensor, receivedStride);

    uiOD->labelInferenceTimeOD->setText(TEXT_INFERENCE + QString("%1 ms").arg(receivedTimeElapsed));

    updateObjectList(outputTensor);

    emit sendMatToView(receivedMat);

    if (continuousMode) {
        /* Stop Total FPS timer and display it to GUI, then restart timer before getting the next frame */
        utilOD->timeTotalFps(false);
        totalFps = utilOD->calculateTotalFps();
        uiOD->labelTotalFps->setText(TEXT_TOTAL_FPS + QString::number(double(totalFps), 'f', 1));
        utilOD->timeTotalFps(true);

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

    for (int i = ITEM_OFFSET; (i + 1) < receivedList.size(); i += 6)
        objectsDetectedList.append(labelList[int(receivedList[i])]);

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
    inputModeOD = cameraMode;

    uiOD->actionLoad_Camera->setEnabled(false);
    uiOD->actionLoad_File->setText(TEXT_LOAD_FILE);
}

void objectDetection::setImageMode()
{
    inputModeOD = imageMode;

    uiOD->actionLoad_Camera->setEnabled(camConnect);
    uiOD->actionLoad_File->setText(TEXT_LOAD_NEW_FILE);
}

void objectDetection::setVideoMode()
{
    inputModeOD = videoMode;

    uiOD->actionLoad_Camera->setEnabled(camConnect);
    uiOD->actionLoad_File->setText(TEXT_LOAD_NEW_FILE);
}

void objectDetection::stopContinuousMode()
{
    continuousMode = false;

    stopVideo();
    setButtonState(true);

    uiOD->labelInferenceTimeOD->setText(TEXT_INFERENCE);
    uiOD->labelTotalFps->setText(TEXT_TOTAL_FPS);
    uiOD->tableWidgetOD->setRowCount(0);

    if (inputModeOD != videoMode)
        emit startVideo();
}
