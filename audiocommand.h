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
#include <sndfile.h>
}

#include <vector>

#include <QGraphicsLineItem>
#include <QMap>
#include <QVector>

class edgeUtils;
class QGraphicsPolygonItem;

namespace Ui { class MainWindow; }

class audioCommand : public QObject
{
    Q_OBJECT
public:
    audioCommand(Ui::MainWindow *ui, QStringList labelFileList, QString inferenceEngine);

    void readAudioFile(QString filePath);

public slots:
    void triggerInference();
    void interpretInference(const QVector<float> &receivedTensor, int receivedTimeElapsed);
    void startListening();

signals:
    void requestInference(void *data, size_t inputDataSize);

private:
    Ui::MainWindow *uiAC;
    edgeUtils *utilAC;
    QStringList labelList;
    bool continuousMode;
    std::vector<float> content;
    QGraphicsPolygonItem *arrow;
    QVector<QGraphicsLineItem*> trail;
    QString history;
    QMap<QString, int> activeCommands;

    QVector<float> sortTensor(const QVector<float> receivedTensor, int receivedStride);
    void updateDetectedWords(QString word);
    void setupArrow();
    void updateArrow(QString instruction);
    void clearTrail();
};

#endif // AUDIOCOMMAND_H
