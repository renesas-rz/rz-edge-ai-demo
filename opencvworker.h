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

#include <linux/v4l2-controls.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <memory>
#include <string.h>

#include <QObject>

#include <linux/types.h>

Q_DECLARE_METATYPE(cv::Mat)

enum Board { G2E, G2L, G2M, Unknown };

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
    void toggleWhitebalanceAuto();
    void toggleGain();
    void toggleExpose();
    void toggleSaturation();

private:
    int runCommand(std::string command, std::string &stdoutput);
    void setControl(__u32 id, __s32 value);
    void setupCamera();
    void connectCamera();
    void checkCamera();

    std::unique_ptr<cv::VideoCapture> videoCapture;
    bool webcamInitialised;
    bool webcamOpened;
    bool usingMipi;
    int connectionAttempts;
    std::string webcamName;
    cv::Mat picture;
    cv::VideoCapture *camera;
    std::string cameraInitialization;
    bool autoWhiteBalance;
    bool autoGain;
    v4l2_exposure_auto_type autoExpose;
};

#endif // OPENCVCAPTUREWORKER_H
