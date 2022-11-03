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

#include "audiocommand.h"
#include "edge-utils.h"

#include "ui_mainwindow.h"

#include <cstring>
#include <math.h>

#include <QGraphicsPolygonItem>
#include <QGraphicsScene>

#define MODEL_NAME_AC "browserfft-speech-renesas.tflite"
#define TEXT_LOAD_AUDIO_FILE "Load Audio File"

#define COMMAND_COL 0
#define COUNT_COL 1

#define BUFFER_SIZE 1024

#define GRID_INC 50
#define GRID_THICKNESS 1

#define ARROW_SIZE 20
#define ARROW_THICKNESS 4

#define ARROW_RIGHT 180
#define ARROW_LEFT 0
#define ARROW_DOWN -90
#define ARROW_UP 90

audioCommand::audioCommand(Ui::MainWindow *ui, QStringList labelFileList, QString inferenceEngine)
{
    uiAC = ui;
    utilAC = new edgeUtils();
    labelList = labelFileList;
    trail = QVector<QGraphicsLineItem*>();
    QGraphicsScene *scene = uiAC->graphicsView->scene();
    QPen pen = QPen(LINE_BLUE);
    int width = uiAC->graphicsView->width();
    int height = uiAC->graphicsView->height();

    pen.setWidth(GRID_THICKNESS);

    /* Create a line grid on the graphics scene */
    for (int x = 0; x <= width; x += GRID_INC)
        scene->addLine(x, 0, x, height, pen);

    for (int y = 0; y <= height; y += GRID_INC)
        scene->addLine(0, y, width, y, pen);

    /* Stacked Widget setup */
    uiAC->stackedWidgetLeft->setCurrentIndex(STACK_WIDGET_INDEX_AC);
    uiAC->stackedWidgetRight->setCurrentIndex(STACK_WIDGET_INDEX_AC);
    uiAC->labelInferenceEngineAC->setText(inferenceEngine);
    uiAC->labelInferenceTimeAC->setText(TEXT_INFERENCE);
    uiAC->labelDemoMode->setText("Mode: Audio Command");
    uiAC->commandReaderAC->clear();

    /* Menu Bar setup */
    uiAC->actionShopping_Basket->setDisabled(false);
    uiAC->actionObject_Detection->setDisabled(false);
    uiAC->actionPose_Estimation->setDisabled(false);
    uiAC->actionFace_Detection->setDisabled(false);
    uiAC->actionAudio_Command->setDisabled(true);
    uiAC->actionLoad_File->setText(TEXT_LOAD_AUDIO_FILE);
    uiAC->actionLoad_Camera->setDisabled(true);

    /* Output Table setup */
    uiAC->tableWidgetAC->verticalHeader()->setDefaultSectionSize(25);
    uiAC->tableWidgetAC->setHorizontalHeaderLabels({"Command", "History"});
    uiAC->tableWidgetAC->horizontalHeader()->setStretchLastSection(true);
    uiAC->tableWidgetAC->verticalHeader()->setStretchLastSection(false);
    uiAC->tableWidgetAC->setEditTriggers(QAbstractItemView::NoEditTriggers);
    uiAC->tableWidgetAC->resizeColumnsToContents();
    double column1Width = uiAC->tableWidgetAC->geometry().width() * 0.5;
    uiAC->tableWidgetAC->setColumnWidth(COMMAND_COL, column1Width);
    uiAC->tableWidgetAC->setColumnWidth(COUNT_COL, column1Width * 0.5);
    uiAC->tableWidgetAC->setRowCount(0);

    uiAC->labelAIModelFilenameAC->setText(MODEL_NAME_AC);
    uiAC->labelInferenceTimePE->setText(TEXT_INFERENCE);
    uiAC->labelTotalFpsPose->setText(TEXT_TOTAL_FPS);

    setupArrow();
}

void audioCommand::setupArrow()
{
    QPen pen = QPen(DOT_RED);
    QBrush brush = QBrush(DOT_RED, Qt::Dense6Pattern);
    QGraphicsScene *scene = uiAC->graphicsView->scene();
    QGraphicsLineItem *line = new QGraphicsLineItem();
    qreal arrowSize = ARROW_SIZE;
    QPolygonF arrowHead = QPolygonF(ARROW_SIZE);
    QPoint gridCentre = QPoint(uiAC->graphicsView->width() / 2, uiAC->graphicsView->height() / 2);

    pen.setWidth(PEN_THICKNESS);

    /* Source for angle, arrowP1 and arrowP2:
     * https://doc.qt.io/qt-5/qtwidgets-graphicsview-diagramscene-example.html */
    double angle = std::atan2(-line->line().dy(), line->line().dx());

    QPointF arrowP1 = line->line().p1() + QPointF(sin(angle + M_PI / 3) * arrowSize,
                                    cos(angle + M_PI / 3) * arrowSize);
    QPointF arrowP2 = line->line().p1() + QPointF(sin(angle + M_PI - M_PI / 3) * arrowSize,
                                    cos(angle + M_PI - M_PI / 3) * arrowSize);

    arrowHead.clear();
    arrowHead << line->line().p1() << arrowP1 << arrowP2;

    arrow = scene->addPolygon(arrowHead, pen, brush);
    arrow->setPos(gridCentre);
    arrow->setRotation(ARROW_UP);
}

void audioCommand::updateArrow(QString instruction)
{
    QGraphicsLineItem *latest;
    QPointF oldPosition = arrow->pos();
    QGraphicsScene *scene = uiAC->graphicsView->scene();
    QPen pen = QPen(DOT_RED);
    QPoint gridCentre = QPoint(uiAC->graphicsView->width() / 2, uiAC->graphicsView->height() / 2);

    pen.setWidth(ARROW_THICKNESS);

    if (instruction.compare("right") == 0) {
        if (arrow->x() < uiAC->graphicsView->width())
            arrow->setPos(arrow->x() + GRID_INC, arrow->y());

        arrow->setRotation(ARROW_RIGHT);
    } else if (instruction.compare("left") == 0) {
        if (arrow->x() > 0)
            arrow->setPos(arrow->x() - GRID_INC, arrow->y());

        arrow->setRotation(ARROW_LEFT);
    } else if (instruction.compare("down") == 0) {
        if (arrow->y() < uiAC->graphicsView->height())
            arrow->setPos(arrow->x(), arrow->y() + GRID_INC);

        arrow->setRotation(ARROW_DOWN);
    } else if (instruction.compare("up") == 0) {
        if (arrow->y() > 0)
            arrow->setPos(arrow->x(), arrow->y() - GRID_INC);

        arrow->setRotation(ARROW_UP);
    } else if (instruction.compare("go") == 0) {
        if (arrow->rotation() == ARROW_RIGHT)
            updateArrow(QString("right"));
        else if (arrow->rotation() == ARROW_LEFT)
            updateArrow(QString("left"));
        else if (arrow->rotation() == ARROW_DOWN)
            updateArrow(QString("down"));
        else if (arrow->rotation() == ARROW_UP)
            updateArrow(QString("up"));

    } else if (instruction.compare("off") == 0 || instruction.compare("stop") == 0) {
        arrow->setPos(gridCentre);
        arrow->setRotation(ARROW_UP);
        clearTrail();

        return;
    }

    latest = scene->addLine(QLineF(oldPosition, arrow->pos()), pen);
    trail.append(latest);
}

void audioCommand::clearTrail()
{
    QGraphicsScene *scene = uiAC->graphicsView->scene();
    QGraphicsLineItem *line;

    foreach (line, trail)
        scene->removeItem(line);

    trail.clear();
}

void audioCommand::interpretInference(const QVector<float> &receivedTensor, int receivedTimeElapsed)
{
    QString label = "Unknown";
    float confidence = DETECT_DEFAULT_THRESHOLD;

    for (int i = 0; i < receivedTensor.size(); i++) {
        float tmp = receivedTensor.at(i);

        if (tmp > confidence) {
            confidence = tmp;
            label = labelList.at(i);
        }
    }

    uiAC->labelInferenceTimeAC->setText(TEXT_INFERENCE + QString("%1 ms").arg(receivedTimeElapsed));

    updateDetectedWords(label);
    updateArrow(label);
}

void audioCommand::triggerInference()
{
    continuousMode = true;

    startListening();
}

void audioCommand::readAudioFile(QString filePath)
{
    char* filePath_c = new char [filePath.length() + 1];
    float data[BUFFER_SIZE];
    sf_count_t readFloats;
    SNDFILE* soundFile;
    SF_INFO metaData;

    content.clear();

    std::strcpy(filePath_c, filePath.toStdString().c_str());

    /* Provide blank metadata when opening the sound file as
     * only the data section is needed */
    soundFile = sf_open(filePath_c, SFM_READ, &metaData);

    readFloats = sf_readf_float(soundFile, data, BUFFER_SIZE);

    while (readFloats > 0) {
        for (int i = 0; i < readFloats; i++)
            content.push_back(data[i]);

        readFloats = sf_readf_float(soundFile, data, BUFFER_SIZE);
    }
}

void audioCommand::startListening()
{
    emit requestInference(content.data(), (size_t) content.size() * sizeof(float));
}

void audioCommand::updateDetectedWords(QString word)
{
    QTableWidget * table = uiAC->tableWidgetAC;
    auto iter = activeCommands.find(word);

    /* Command table update */
    if (iter == activeCommands.end()) {
        QTableWidgetItem* wordItem = new QTableWidgetItem(word);
        QTableWidgetItem* totalItem = new QTableWidgetItem("1");

        wordItem->setTextAlignment(Qt::AlignCenter);
        totalItem->setTextAlignment(Qt::AlignCenter);

        table->insertRow(table->rowCount());
        table->setItem(table->rowCount() - 1, COMMAND_COL, wordItem);
        table->setItem(table->rowCount() - 1, COUNT_COL, totalItem);

        activeCommands.insert(word, 1);
    } else {
        int occurances = iter.value() + 1;

        activeCommands.remove(iter.key());
        activeCommands.insert(word, occurances);

        for (int i = 0; i < table->rowCount(); i++)
            if (table->item(i, 0)->text().compare(word) == 0)
                table->item(i, 1)->setText(QString::number(occurances));
    }

    uiAC->tableWidgetAC->scrollToBottom();

    /* Clear history on keyword "off" or "stop" */
    if (word.compare("off") == 0 || word.compare("stop") == 0) {
        history.clear();
        activeCommands.clear();
        table->setRowCount(0);
    } else {
        history.append(" " + word);
    }

    /* Active commands update */
    uiAC->commandReaderAC->setText(history);
}
