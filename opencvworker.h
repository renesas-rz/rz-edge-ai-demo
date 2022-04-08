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

#define G2L_CAM_INIT "media-ctl -d /dev/media0 --reset && media-ctl -d /dev/media0 -l \"'rzg2l_csi2 10830400.csi2':1->'CRU output':0 [1]\" && media-ctl -d /dev/media0 -V \"'rzg2l_csi2 10830400.csi2':1 [fmt:UYVY8_2X8/800x600 field:none]\" && media-ctl -d /dev/media0 -V \"'ov5645 0-003c':0 [fmt:UYVY8_2X8/800x600 field:none]\""
#define G2M_CAM_INIT "media-ctl -d /dev/media0 -r && media-ctl -d /dev/media0 -l \"'rcar_csi2 fea80000.csi2':1->'VIN0 output':0 [1]\" && media-ctl -d /dev/media0 -V \"'rcar_csi2 fea80000.csi2':1 [fmt:UYVY8_2X8/800x600 field:none]\" && media-ctl -d /dev/media0 -V \"'ov5645 2-003c':0 [fmt:UYVY8_2X8/800x600 field:none]\""
#define G2E_CAM_INIT "media-ctl -d /dev/media0 -r && media-ctl -d /dev/media0 -l \"'rcar_csi2 feaa0000.csi2':1->'VIN4 output':0 [1]\" && media-ctl -d /dev/media0 -V \"'rcar_csi2 feaa0000.csi2':1 [fmt:UYVY8_2X8/800x600 field:none]\" && media-ctl -d /dev/media0 -V \"'ov5645 3-003c':0 [fmt:UYVY8_2X8/800x600 field:none]\""

#define GST_CODEC_PIPELINE " ! qtdemux ! queue ! h264parse ! omxh264dec ! queue ! vspmfilter dmabuf-use=true ! "
#define GST_NO_CODEC_PIPELINE " ! decodebin ! videoscale ! videoconvert ! "

#define VIDEO_ASPECT_RATIO_4_TO_3 1.33
#define VIDEO_ASPECT_RATIO_5_TO_4 1.25
#define VIDEO_ASPECT_RATIO_16_TO_9 1.78
#define VIDEO_ASPECT_RATIO_16_TO_10 1.6

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <string.h>

#include <QObject>

Q_DECLARE_METATYPE(cv::Mat)

enum Board { G2E, G2L, G2LC, G2M, Unknown };
enum InputOpenCV { cameraInput, imageInput, videoInput };

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
    void useVideoMode(QString videoFilePath);

private slots:
    void getVideoFileFrame();

private:
    int runCommand(std::string command, std::string &stdoutput);
    void setupCamera();
    void connectCamera();
    void checkCamera();
    void checkVideoFile();
    void setVideoDims();

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
    cv::VideoCapture *camera;
    cv::Mat imageFile;
    cv::VideoCapture *videoFile;
    std::string cameraInitialization;
    InputOpenCV inputOpenCV;
};

#endif // OPENCVCAPTUREWORKER_H
