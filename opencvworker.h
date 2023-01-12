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

#ifndef OPENCVCAPTUREWORKER_H
#define OPENCVCAPTUREWORKER_H

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "edge-utils.h"

#include <string.h>

#include <QObject>

Q_DECLARE_METATYPE(cv::Mat)

class opencvWorker : public QObject
{
    Q_OBJECT

public:
    opencvWorker(QString cameraLocation, Board board);
    ~opencvWorker();
    cv::Mat* getImage(unsigned int iterations);
    bool cameraInit();
    bool getCameraOpen();
    bool getUsingMipi();
    void useCameraMode();
    void useImageMode(QString imageFilePath);
    bool useVideoMode(QString videoFilePath);

signals:
    void resolutionError(QString message);

private slots:
    void getVideoFileFrame();

private:
    int runCommand(std::string command, std::string &stdoutput);
    void setupCamera();
    void connectCamera();
    void checkCamera();
    void checkVideoFile();
    bool setVideoDims();

    std::unique_ptr<cv::VideoCapture> videoCapture;
    bool webcamInitialised;
    bool webcamOpened;
    bool usingMipi;
    bool videoCodecs;
    int videoHeight;
    int videoWidth;
    int connectionAttempts;
    QString imagePath;
    QString videoLoadedPath;
    std::string webcamName;
    cv::Mat picture;
    cv::VideoCapture camera;
    cv::Mat imageFile;
    cv::VideoCapture *videoFile;
    std::string cameraInitialization;
    Input inputOpenCV;
};

#endif // OPENCVCAPTUREWORKER_H
