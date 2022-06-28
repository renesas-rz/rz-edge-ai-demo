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

#ifndef OBJECTDETECTION_H
#define OBJECTDETECTION_H

#include <QMainWindow>

#include <opencv2/videoio.hpp>

#include "edge-utils.h"

#define TEXT_LOAD_FILE "Load Image/Video"
#define TEXT_LOAD_NEW_FILE "Load New Image/Video"

class QGraphicsScene;
class edgeUtils;

namespace Ui { class MainWindow; }

class objectDetection : public QObject
{
     Q_OBJECT

public:
    objectDetection(Ui::MainWindow *ui, QStringList labelFileList);
    void setImageMode();
    void setVideoMode();
    void setCameraMode();

public slots:
    void runInference(QVector<float> receivedTensor, int receivedStride, int receivedTimeElapsed, const cv::Mat&receivedMat);

signals:
    void getFrame();
    void getBoxes(const QVector<float>& receivedTensor, QStringList labelList);
    void sendMatToView(const cv::Mat&receivedMat);
    void startVideo();
    void stopVideo();

private slots:
    void stopContinuousMode();
    void triggerInference();

private:
    void setButtonState(bool enable);
    QVector<float> sortTensor(QVector<float> &receivedTensor, int receivedStride);
    void updateObjectList(const QVector<float> receivedList);

    Ui::MainWindow *uiOD;
    edgeUtils *utilOD;
    QVector<float> outputTensor;
    bool buttonState;
    bool continuousMode;
    QStringList labelList;
    Input inputModeOD;
};

#endif // OBJECTDETECTION_H
