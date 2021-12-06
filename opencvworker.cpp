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

#include <QDebug>

#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "opencvworker.h"

#include <opencv2/imgproc/imgproc.hpp>

opencvWorker::opencvWorker(QString cameraLocation, Board board)
{
    webcamName = cameraLocation.toStdString();
    connectionAttempts = 0;

    setupCamera();

    if (usingMipi) {
        if (board == G2M)
            cameraInitialization = G2M_CAM_INIT;
        else if (board == G2E)
            cameraInitialization = G2E_CAM_INIT;
        else if (board == G2L)
            cameraInitialization = G2L_CAM_INIT;
    }

    connectCamera();

    /* These settings are inverted by calls to the toggle slots below,
     * ensuring we have a suitable set of defaults for the sensor */
    autoWhiteBalance = false;
    autoGain = false;
    autoExpose = V4L2_EXPOSURE_MANUAL;

    if (usingMipi) {
        toggleWhitebalanceAuto();
        toggleGain();
        toggleExpose();
    }
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

void opencvWorker::setupCamera()
{
    struct v4l2_capability cap;
    int fd = open(webcamName.c_str(), O_RDONLY);

    if (fd == -1)
        qWarning() << "Could not open file:" << webcamName.c_str();

    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
        qWarning("Could not retrieve v4l2 camera information");
        webcamInitialised = false;
    } else {
        webcamInitialised = true;

        std::string busInfo = std::string((char*)cap.bus_info);

        if (busInfo.find("platform") != std::string::npos) {
            usingMipi = true;
        } else if (busInfo.find("usb") != std::string::npos) {
            usingMipi = false;
        } else {
            qWarning("Camera format error, defaulting to MIPI");
            usingMipi = true;
        }
    }
    close(fd);
}

void opencvWorker::setControl(__u32 id, __s32 value)
{

    if (usingMipi) {
        struct v4l2_control control;
        int fd = open("/dev/v4l-subdev1", O_RDWR);

        if (fd <= 0) {
            qWarning("Warning: Failed to open node (/dev/v4l-subdev1)");
            return;
        }

        control.id = id;
        control.value = value;

        if (ioctl(fd, VIDIOC_S_CTRL, &control) == -1)
            qWarning() << "VIDIOC_S_CTRL, errno:" << errno;

        close(fd);
    }
}

void opencvWorker::toggleWhitebalanceAuto()
{
    autoWhiteBalance = !autoWhiteBalance;
    setControl(V4L2_CID_AUTO_WHITE_BALANCE, autoWhiteBalance);
}

void opencvWorker::toggleGain()
{
    autoGain = !autoGain;
    setControl(V4L2_CID_AUTOGAIN, autoGain);
}

void opencvWorker::toggleExpose()
{
    if (autoExpose == V4L2_EXPOSURE_MANUAL)
        autoExpose = V4L2_EXPOSURE_AUTO;
    else
        autoExpose = V4L2_EXPOSURE_MANUAL;

    setControl(V4L2_CID_EXPOSURE_AUTO, autoExpose);
}

void opencvWorker::connectCamera()
{
    connectionAttempts++;
    int cameraWidth = 800;
    int cameraHeight = 600;

    if (usingMipi) {
        std::string stdoutput;

        /* Run media-ctl command */
        if (runCommand(cameraInitialization, stdoutput))
            qWarning("Cannot initialize the camera");
    }

    /* Define the format for the camera to use */
    camera = new cv::VideoCapture(webcamName);
    camera->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('U', 'Y', 'V', 'Y'));
    camera->open(webcamName);

    if (!camera->isOpened()) {
        qWarning("Cannot open the camera");
        webcamOpened = false;
    } else {
        webcamOpened = true;
    }

    if (!usingMipi) {
        camera->set(cv::CAP_PROP_FPS, 10);
        camera->set(cv::CAP_PROP_BUFFERSIZE, 1);
        cameraWidth = 1280;
        cameraHeight = 720;
    }

    camera->set(cv::CAP_PROP_FRAME_WIDTH, cameraWidth);
    camera->set(cv::CAP_PROP_FRAME_HEIGHT, cameraHeight);

    checkCamera();
}

void opencvWorker::checkCamera()
{
    /* Check to see if camera can retrieve a frame*/
    *camera >> picture;

    if (picture.empty()) {
        qWarning("Lost connection to camera, reconnecting");
        camera->release();

        if (connectionAttempts < 3) {
            connectCamera();
        } else {
            qWarning() << "Could not retrieve a frame, attempts: " << connectionAttempts;
            webcamInitialised = false;
        }
    }
}

opencvWorker::~opencvWorker() {
    camera->release();
}

cv::Mat* opencvWorker::getImage(unsigned int iterations)
{
    do {
        *camera >> picture;

        if (picture.empty()) {
            qWarning("Image retrieval error");
            return nullptr;
        }

    } while (--iterations);

    cv::cvtColor(picture, picture, cv::COLOR_BGR2RGB);

    return &picture;
}

bool opencvWorker::getUsingMipi()
{
    return usingMipi;
}

bool opencvWorker::cameraInit()
{
    return webcamInitialised;
}

bool opencvWorker::getCameraOpen()
{
    return  webcamOpened;
}
