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

#include <QCamera>
#include <QCameraImageCapture>

#include "opencvworker.h"

#include <opencv2/imgproc/imgproc.hpp>


opencvWorker::opencvWorker(QString cameraLocation)
{
    webcamName = cameraLocation;

    usingMipi = true;

    if (usingMipi) {
        std::string stdoutput;

        /* We need to run this command only once */
        if (runCommand(cameraInitialization, stdoutput))
            qWarning("Cannot initialize the camera");
    }

    /* Define the format for the camera to use */
    camera = new cv::VideoCapture(webcamName.toStdString());
    camera->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('U', 'Y', 'V', 'Y'));
    camera->open(webcamName.toStdString());

    if (!camera->isOpened())
        qWarning("Cannot open the camera");

    camera->set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    camera->set(cv::CAP_PROP_FRAME_HEIGHT, 960);

    picture = cv::Mat();
}

int opencvWorker::runCommand(std::string command, std::string &stdoutput)
{
    size_t charsRead;
    int status;
    char buffer[512];

    FILE *output = popen(command.c_str(), "r");

    if (output == NULL)
        qWarning("cannot execute command");

    stdoutput = "";
    while (true) {
        charsRead = fread(buffer, sizeof(char), sizeof(buffer), output);

        if (charsRead > 0)
            stdoutput += std::string(buffer, charsRead);

        if (ferror(output)) {
            pclose(output);
            qWarning("Could not retreive output from command");
        }

        if (feof(output))
            break;
    }

    status = pclose(output);
    if (status == -1)
        qWarning("Could not terminate from command");

    return WEXITSTATUS(status);
}


opencvWorker::~opencvWorker() {
    camera->release();
}

cv::Mat* opencvWorker::getImage()
{
    int iterations = 6;
    do {
        *camera >> picture;

        if (picture.empty())
            qWarning("Image retrieval error");

    } while (--iterations);

    return &picture;
}

void opencvWorker::initialiseWebcam(QString cameraLocation)
{
    QString gstreamerPipeline;

    getResolution();
    if (webcamInitialised == true)
        videoCapture->release();

    if (!cameraLocation.isEmpty()) {
        gstreamerPipeline = "v4l2src device=" + QString(cameraLocation) +
            " ! video/x-raw, width=" + QString::number(cameraWidth) + ", height=" + QString::number(cameraHeight) +
            " ! videoconvert ! appsink";

        videoCapture.reset(new cv::VideoCapture(gstreamerPipeline.toStdString(), cv::CAP_GSTREAMER));

        if (videoCapture->isOpened())
            webcamInitialised = true;
        else
            webcamInitialised = false;

    } else {
        webcamInitialised = false;
    }

    emit webcamInit(webcamInitialised);
}

void opencvWorker::getResolution()
{
    QList<QByteArray> cameraDevices = QCamera::availableDevices();
    QList<QSize> resolutions;
    QCamera *camera;
    QCameraImageCapture *imageCapture;

    if (cameraDevices.count() > 0) {
        QByteArray cameraDevice = cameraDevices.first();
        camera = new QCamera(cameraDevice);
        camera->load();
        imageCapture = new QCameraImageCapture(camera);
        resolutions = imageCapture->supportedResolutions();

        QSize resol = resolutions.takeLast();
        cameraWidth = resol.width();
        cameraHeight = resol.height();
    }
}

void opencvWorker::readFrame()
{
    if(!videoCapture->read(videoFrame)) {
        webcamInitialised = false;
        videoCapture->release();
        emit webcamInit(webcamInitialised);
        return;
    }

    emit sendImage(videoFrame);
}

void opencvWorker::checkWebcam()
{
    if(!videoCapture->read(videoFrame)) {
        webcamInitialised = false;
        videoCapture->release();
        return;
    } else
         webcamInitialised = true;
}

void opencvWorker::disconnectWebcam()
{
    if (webcamInitialised == true)
        videoCapture->release();
}
