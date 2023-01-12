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

enum Delegate { armNN, xnnpack, none };

class tfliteWorker : public QObject
{
    Q_OBJECT

public:
    tfliteWorker(QString modelLocation, Delegate armnnDelegate, int defaultThreads);
    ~tfliteWorker();
    void receiveImage(const cv::Mat&);
    void setDemoMode(Mode demoMode);

public slots:
    void processData(void *data, size_t dataSize);

signals:
    void sendOutputTensor(const QVector<float>&, int, int, const cv::Mat&);
    void sendOutputTensorImageless(const QVector<float>&, int, int);
    void sendOutputTensorBasic(const QVector<float>&, int);
    void sendInferenceWarning(QString warningMessage);

private:
    std::unique_ptr<tflite::Interpreter> tfliteInterpreter;
    std::unique_ptr<tflite::FlatBufferModel> tfliteModel;
    QString modelName;
    Delegate delegateType;
    Mode modeSelected;
    TfLiteDelegate* xnnpack_delegate;
    QVector<float> outputTensor;
    const cv::Mat *displayMat;
    int wantedWidth, wantedHeight, wantedChannels;
};

#endif // TFLITEWORKER_H
