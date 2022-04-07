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

#include <chrono>
#include <opencv2/videoio.hpp>

#define BUTTON_BLUE "background-color: rgba(42, 40, 157);color: rgb(255, 255, 255);border: 2px;border-radius: 55px;border-style: outset;"
#define BUTTON_RED "background-color: rgba(255, 0, 0);color: rgb(255, 255, 255);border: 2px;border-radius: 55px;border-style: outset;"

#define TEXT_INFERENCE "Inference Time: "
#define TEXT_LOAD_FILE "Load Image/Video"
#define TEXT_LOAD_NEW_FILE "Load New Image/Video"
#define TEXT_TOTAL_FPS "Total FPS: "

class QGraphicsScene;

namespace Ui { class MainWindow; }

enum InputOD { cameraModeOD, imageModeOD, videoModeOD };

class objectDetection : public QObject
{
     Q_OBJECT

public:
    objectDetection(Ui::MainWindow *ui, const QString labelPath);
    void setImageMode();
    void setVideoMode();
    void setCameraMode();

public slots:
    void runInference(const QVector<float>& receivedTensor, int receivedTimeElapsed, const cv::Mat&receivedMat);

signals:
    void getFrame();
    void getBoxes(const QVector<float>& receivedTensor, QStringList labelList);
    void restartVideo();
    void sendMatToView(const cv::Mat&receivedMat);
    void setPlayIcon(bool state);
    void startVideo();
    void stopVideo();

private slots:
    void stopContinuousMode();
    void triggerInference();

private:
    void setButtonState(bool enable);
    void displayTotalFps(int totalProcessTime);
    void timeTotalFps(bool startingTimer);
    void updateObjectList(const QVector<float> receivedList);

    Ui::MainWindow *uiOD;
    QVector<float> outputTensor;
    bool buttonState;
    bool continuousMode;
    QStringList labelList;
    std::chrono::high_resolution_clock::time_point startTime;
    InputOD inputModeOD;
};

#endif // OBJECTDETECTION_H
