/*****************************************************************************************
 * Copyright (C) 2023 Renesas Electronics Corp.
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

#ifndef AUDIOCOMMAND_H
#define AUDIOCOMMAND_H

extern "C" {
#include <alsa/asoundlib.h>
#include <sndfile.h>
}

#include <vector>
#include <thread>

#include <QGraphicsLineItem>
#include <QMap>
#include <QVector>

#include "edge-utils.h"

class QGraphicsPolygonItem;

namespace Ui { class MainWindow; }

class audioCommand : public QObject
{
    Q_OBJECT
public:
    audioCommand(Ui::MainWindow *ui, QStringList labelFileList, QString inferenceEngine);
    ~audioCommand();

    void readAudioFile(QString filePath);
    void setMicMode();

public slots:
    void interpretInference(const QVector<float> &receivedTensor, int receivedTimeElapsed);
    void toggleAudioInput();

signals:
    void requestInference(void *data, size_t inputDataSize);
    void micWarning(QString message);

private:
    Ui::MainWindow *uiAC;
    edgeUtils *utilAC;
    QStringList labelList;
    bool buttonIdleBlue;
    bool firstBlock;
    bool recordButtonMutex;
    std::vector<float> content;
    QGraphicsPolygonItem *arrow;
    QVector<QGraphicsLineItem*> trail;
    QString history;
    QMap<QString, int> activeCommands;
    float ambiantVol;
    snd_pcm_t *mic_pcm;
    snd_pcm_hw_params_t *hw_params;
    unsigned int sampleRate;
    Input inputModeAC;
    std::thread secThread;

    QVector<float> sortTensor(const QVector<float> receivedTensor, int receivedStride);
    void updateDetectedWords(QString word);
    void setupArrow();
    void updateArrow(QString instruction);
    void toggleTalkButtonState();
    void startListening();
    bool setupMic();
    void closeMic();
    void clearTrail();
    bool recordSecond();
    bool checkForVolIncrease();
};

#endif // AUDIOCOMMAND_H
