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

#ifndef TFLITEWORKER_H
#define TFLITEWORKER_H

#include <tensorflow/lite/kernels/register.h>

#include "edge-utils.h"

#include <QObject>
#include <QVector>

#include <opencv2/videoio.hpp>

#define WARNING_IMAGE_RETREIVAL "Received invalid image path, could not run inference"
#define WARNING_INVOKE "Failed to run invoke"
#define WARNING_UNSUPPORTED_DATA_TYPE "Model data type currently not supported"

#define SCALE_FACTOR_UCHAR_TO_FLOAT (1/255.0F)

enum Delegate { armNN, xnnpack, none };

class tfliteWorker : public QObject
{
    Q_OBJECT

public:
    tfliteWorker(QString modelLocation, Delegate armnnDelegate, int defaultThreads);
    ~tfliteWorker();
    void receiveImage(const cv::Mat&);
    void setDemoMode(Mode demoMode);

signals:
    void sendOutputTensor(const QVector<float>&, int, int, const cv::Mat&);
    void sendOutputTensorImageless(const QVector<float>&, int, int);
    void sendInferenceWarning(QString warningMessage);

private:
    std::unique_ptr<tflite::Interpreter> tfliteInterpreter;
    std::unique_ptr<tflite::FlatBufferModel> tfliteModel;
    QString modelName;
    Delegate delegateType;
    Mode modeSelected;
#ifdef DUNFELL
    TfLiteDelegate* xnnpack_delegate;
#endif
    QVector<float> outputTensor;
    int wantedWidth, wantedHeight, wantedChannels;
};

#endif // TFLITEWORKER_H
