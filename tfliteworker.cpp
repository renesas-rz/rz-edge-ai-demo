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

#include <chrono>

#include "tfliteworker.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <armnn/ArmNN.hpp>
#include <armnn/Utils.hpp>
#include <delegate/armnn_delegate.hpp>
#include <delegate/DelegateOptions.hpp>
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

#define WARNING_IMAGE_RETREIVAL "Received invalid image path, could not run inference"
#define WARNING_INVOKE "Failed to run invoke"
#define WARNING_UNSUPPORTED_DATA_TYPE "Model data type currently not supported"

#define SCALE_FACTOR_UCHAR_TO_FLOAT (1/255.0F)

tfliteWorker::tfliteWorker(QString modelLocation, Delegate delegateType, int defaultThreads)
{
    tflite::ops::builtin::BuiltinOpResolver tfliteResolver;
    TfLiteIntArray *wantedDimensions;
    this->delegateType = delegateType;
    modelName = modelLocation;

    tfliteModel = tflite::FlatBufferModel::BuildFromFile(modelLocation.toStdString().c_str());
    tflite::InterpreterBuilder(*tfliteModel, tfliteResolver) (&tfliteInterpreter);

    /* Setup the delegate */
    if(delegateType == armNN) {
        std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
        armnnDelegate::DelegateOptions delegateOptions(backends);
        std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
            armnnTfLiteDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
            armnnDelegate::TfLiteArmnnDelegateDelete);

        /* Instruct the Interpreter to use the armnnDelegate */
        if (tfliteInterpreter->ModifyGraphWithDelegate(std::move(armnnTfLiteDelegate)) != kTfLiteOk)
           qWarning("ArmNN Delegate could not be used to modify the graph\n");
    }

    if (delegateType == xnnpack) {
        TfLiteXNNPackDelegateOptions xnnpack_options = TfLiteXNNPackDelegateOptionsDefault();

        xnnpack_options.num_threads = defaultThreads;
        xnnpack_delegate = TfLiteXNNPackDelegateCreate(&xnnpack_options);

        if (tfliteInterpreter->ModifyGraphWithDelegate(xnnpack_delegate) != kTfLiteOk)
            qWarning("Could not modifiy Graph with XNNPack Delegate\n");
    }

    if (tfliteInterpreter->AllocateTensors() != kTfLiteOk)
        qFatal("Failed to allocate tensors!");

    tfliteInterpreter->SetProfiler(nullptr);
    tfliteInterpreter->SetNumThreads(defaultThreads);

    wantedDimensions = tfliteInterpreter->tensor(tfliteInterpreter->inputs()[0])->dims;
    wantedHeight = wantedDimensions->data[1];
    wantedWidth = wantedDimensions->data[2];
    wantedChannels = wantedDimensions->data[3];
}

tfliteWorker::~tfliteWorker() {
    tfliteInterpreter.reset();

    if (delegateType == xnnpack)
        TfLiteXNNPackDelegateDelete(xnnpack_delegate);
}

/* Resize the input image and manipulate the data such that the alpha channel
 * is removed. Input the data to the tensor and output the results into a vector.
 * Also measure the time it takes for this function to complete */
void tfliteWorker::receiveImage(const cv::Mat& sentMat)
{
    cv::Mat sentImageMat;
    int input;

    if(sentMat.empty()) {
        qWarning(WARNING_IMAGE_RETREIVAL);
        emit sendInferenceWarning(WARNING_IMAGE_RETREIVAL);
        return;
    }

    displayMat = &sentMat;

    input = tfliteInterpreter->inputs()[0];

    cv::resize(sentMat, sentImageMat, cv::Size(wantedWidth, wantedHeight));

    if (tfliteInterpreter->tensor(input)->type == kTfLiteFloat32) {
        /* Convert cv::Mat data type from 8-bit unsigned char to 32-bit float.
         * The data of the image needs to be divided by 255.0f as CV_8UC3 ranges
         * from 0 to 255, whereas CV_32FC3 ranges from 0 to 1 */
        sentImageMat.convertTo(sentImageMat, CV_32FC3, SCALE_FACTOR_UCHAR_TO_FLOAT);
    }

    processData(sentImageMat.data, sentImageMat.total() * sentImageMat.elemSize());
}

void tfliteWorker::processData(void *data, size_t inputDataSize)
{
    std::chrono::high_resolution_clock::time_point startTime, stopTime;
    QVector<int> outputTensorCount;
    int timeElapsed;
    int itemStride;
    int input = tfliteInterpreter->inputs()[0];

    if (tfliteInterpreter->tensor(input)->type == kTfLiteFloat32) {
        memcpy(tfliteInterpreter->typed_tensor<float>(input), data, inputDataSize);
    } else if (tfliteInterpreter->tensor(input)->type == kTfLiteUInt8) {
        memcpy(tfliteInterpreter->typed_tensor<uint8_t>(input), data, inputDataSize);
    } else {
        qWarning("Model data type currently not supported!");
        emit sendInferenceWarning(WARNING_UNSUPPORTED_DATA_TYPE);
        return;
    }

    startTime = std::chrono::high_resolution_clock::now();

    if (tfliteInterpreter->Invoke() != kTfLiteOk) {
        stopTime = std::chrono::high_resolution_clock::now();

        qWarning(WARNING_INVOKE);
        emit sendInferenceWarning(WARNING_INVOKE);
        return;
    }

    stopTime = std::chrono::high_resolution_clock::now();

    /* Cycle through each output tensor and store all data */
    for (size_t i = 0; i < tfliteInterpreter->outputs().size(); i++) {
        size_t dataSize = sizeof(float);

        if (tfliteInterpreter->output_tensor(i)->type == kTfLiteFloat32)
            dataSize = sizeof(float);
        else if (tfliteInterpreter->output_tensor(i)->type == kTfLiteUInt8)
            dataSize = sizeof(uint8_t);

        /* Total number of data elements */
        int outputCount = tfliteInterpreter->output_tensor(i)->bytes / dataSize;

        outputTensorCount.push_back(outputCount);

        for (int k = 0; k < outputCount; k++) {
                float output = tfliteInterpreter->typed_output_tensor<float>(i)[k];
                outputTensor.push_back(output);
        }
    }

    /* Set the item stride based on demo mode being used */
    if (modeSelected == PE) {
        itemStride = outputTensorCount.takeFirst();
    } else {
        /* The final output is unused for object detection/recognition models */
        outputTensorCount.removeLast();

        itemStride = outputTensorCount.takeLast();
    }

    timeElapsed = int(std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime).count());

    if (modeSelected == FD && modelName != MODEL_PATH_FD_FACE_DETECTION)
        emit sendOutputTensorImageless(outputTensor, itemStride, timeElapsed);
    else if (modeSelected == AC)
        emit sendOutputTensorBasic(outputTensor, timeElapsed);
    else
        emit sendOutputTensor(outputTensor, itemStride, timeElapsed, *displayMat);

    outputTensor.clear();
}

void tfliteWorker::setDemoMode(Mode demoMode)
{
    modeSelected = demoMode;
}
