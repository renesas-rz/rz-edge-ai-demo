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

#include <QDebug>

#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <math.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "opencvworker.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

opencvWorker::opencvWorker(QString cameraLocation, Board board)
{
    webcamName = cameraLocation.toStdString();
    connectionAttempts = 0;
    inputOpenCV = cameraMode;
    videoCodecs = true;

    setupCamera();

    if (webcamInitialised && usingMipi) {
        if (board == G2M)
            cameraInitialization = G2M_CAM_INIT;
        else if (board == G2E)
            cameraInitialization = G2E_CAM_INIT;
        else if (board == G2L || board == G2LC)
            cameraInitialization = G2L_CAM_INIT;
    }

    if (board == G2LC)
        videoCodecs = false;

    if (webcamInitialised)
        connectCamera();
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
    camera = new cv::VideoCapture(webcamName, cv::CAP_V4L2);
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
    videoFile->release();
}

cv::Mat* opencvWorker::getImage(unsigned int iterations)
{
    if (inputOpenCV == imageMode) {
        /* For image file input, read image from file */
        picture = cv::imread(imagePath.toStdString());
    } else if (inputOpenCV == videoMode) {
        /* For video file input, grab the current frame from the video playback device */
        getVideoFileFrame();

        if (picture.empty()) {
            qWarning("Video frame retrieval error");
            return nullptr;
        }
    } else {
        /* For camera input, grab the latest frame from the camera */
        do {
            *camera >> picture;

            if (picture.empty()) {
                qWarning("Image retrieval error");
                return nullptr;
            }

        } while (--iterations);
    }
    cv::cvtColor(picture, picture, cv::COLOR_BGR2RGB);

    return &picture;
}

void opencvWorker::getVideoFileFrame()
{
    if (videoFile == 0)
        return;

    int prevFramePos = videoFile->get(cv::CAP_PROP_POS_FRAMES);

    *videoFile >> picture;

    /* Set the position of the video back to the start when it reaches the end */
    if (videoFile->get(cv::CAP_PROP_POS_FRAMES) == prevFramePos) {
        qWarning("Reached end of video, restarted playback");
        videoFile->release();
        useVideoMode(videoLoadedPath);
        *videoFile >> picture;
    }
}

void opencvWorker::useImageMode(QString imageFilePath)
{
    checkVideoFile();
    inputOpenCV = imageMode;
    imagePath = imageFilePath;
}

void opencvWorker::useCameraMode()
{
    checkVideoFile();
    inputOpenCV = cameraMode;
}

bool opencvWorker::useVideoMode(QString videoFilePath)
{
    QString videoDecodePipeline;

    checkVideoFile();
    inputOpenCV = videoMode;
    videoLoadedPath = videoFilePath;

    if (!setVideoDims()) {
        videoLoadedPath = "";
        videoFile = 0;

        return false;
    }

    /* Check which decoding pipeline to use */
    if (videoCodecs)
        videoDecodePipeline = GST_CODEC_PIPELINE;
    else
        videoDecodePipeline = GST_NO_CODEC_PIPELINE;

    QString videoPipeline = "filesrc location=" + QString(videoLoadedPath) + QString(videoDecodePipeline) +
            " video/x-raw, format=BGR, width=" + QString::number(videoWidth) + ", height=" + QString::number(videoHeight) + " ! appsink";

    videoFile = new cv::VideoCapture(videoPipeline.toStdString(), cv::CAP_GSTREAMER);

    if (!videoFile->isOpened()) {
        qWarning("Could not open video file for streaming");
        videoFile->release();
        videoFile = 0;
        emit resolutionError(STREAM_OPEN_ERR);

        return false;
    }

    return true;
}

void opencvWorker::checkVideoFile()
{
    /* Close video file capture device */
    if (inputOpenCV == videoMode && videoFile)
        videoFile->release();
}

bool opencvWorker::setVideoDims()
{
    cv::VideoCapture *videoChecker = new cv::VideoCapture(videoLoadedPath.toStdString());

    if (!videoChecker->isOpened()) {
        qWarning("Could not open video file for dimension reading");
        emit resolutionError(FILE_OPEN_ERR);

        return false;
    }

    double videoFileHeight = videoChecker->get(cv::CAP_PROP_FRAME_HEIGHT);
    double videoFileWidth = videoChecker->get(cv::CAP_PROP_FRAME_WIDTH);
    double aspectRatio = videoFileWidth / videoFileHeight;

    videoChecker->release();

    if (videoFileHeight == 0 || videoFileWidth == 0) {
        qWarning("File resolution error.");
        emit resolutionError(RESOLUTION_ERR);

        return false;
    }

    /* Round aspect ratio to 2 decimal places */
    double aspectRatioRounded = round(aspectRatio * 100) / 100;

    /* Identify aspect ratio and set downscaled resolution.
     * Downscaled resolutions set must allign with 32 to prevent any display issues */
    if ((aspectRatioRounded == VIDEO_ASPECT_RATIO_16_TO_9) || (aspectRatioRounded == VIDEO_ASPECT_RATIO_16_TO_10)) {
        videoHeight = 480;
        videoWidth = 768;
    } else if (aspectRatioRounded == VIDEO_ASPECT_RATIO_4_TO_3) {
        videoHeight = 576;
        videoWidth = 768;
    } else if (aspectRatioRounded == VIDEO_ASPECT_RATIO_5_TO_4) {
        videoHeight = 512;
        videoWidth = 640;
    } else {
        videoHeight = videoFileHeight;
        videoWidth = videoFileWidth;
    }

    return true;
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
