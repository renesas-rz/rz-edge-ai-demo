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

#include <QDebug>
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

#define BACKGROUND_OVERHEAD 2 // How many times louder we expect speech to be than ambiant

#define MIC_CHANNELS 1
#define MIC_DEVICE "plughw:0,0"

#define MODEL_SAMPLE_RATE 44100
#define AUDIO_DETECT_THRESHOLD 0.80

#define MIC_OPEN_WARNING "Warning: Cannot open audio device"
#define MIC_HW_PARAM_ALLOC_WARNING "Warning: Cannot allocate hardware parameter structure"
#define MIC_HW_DETAILS_WARNING "Warning: Cannot setup hardware details structure"
#define MIC_ACCESS_TYPE_WARNING "Warning: Cannot hardware set access type"
#define MIC_SAMPLE_FORMAT_WARNING "Warning: Cannot set sample format"
#define MIC_SAMPLE_RATE_WARNING "Warning: Cannot set sample rate"
#define MIC_CHANNEL_WARNING "Warning: Cannot set channel count"
#define MIC_PARAMETER_SET_WARNING "Warning: Cannot set parameters"
#define MIC_PREPARE_WARNING "Warning: Cannot prepare audio interface for use"

#define READ_OVERRUN_ERROR "Error: Overrun occurred during microphone read"
#define READ_PCM_ERROR "Error: Bad PCM State"
#define READ_SUSPEND_ERROR "Error: Suspend event occurred"
#define READ_INCOMPLETE_WARNING "Warning: Incomplete read from microphone"


audioCommand::audioCommand(Ui::MainWindow *ui, QStringList labelFileList, QString inferenceEngine)
{
    uiAC = ui;
    utilAC = new edgeUtils();
    labelList = labelFileList;
    trail = QVector<QGraphicsLineItem*>();
    QGraphicsScene *scene = uiAC->graphicsView->scene();
    QPen pen = QPen(THEME_BLUE);
    int width = uiAC->graphicsView->width();
    int height = uiAC->graphicsView->height();
    inputModeAC = micMode;
    buttonIdleBlue = true;
    firstBlock = true;
    ambiantVol = 0;
    sampleRate = MODEL_SAMPLE_RATE;
    recordButtonMutex = false;

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
    uiAC->actionLoad_Periph->setDisabled(true);
    uiAC->actionLoad_Periph->setText(TEXT_LOAD_MIC);

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
    QPen pen = QPen(THEME_RED);
    QBrush brush = QBrush(THEME_RED, Qt::Dense6Pattern);
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
    QPen pen = QPen(THEME_RED);
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

    } else if (instruction.compare("off") == 0) {
        arrow->setPos(gridCentre);
        arrow->setRotation(ARROW_UP);
        clearTrail();
        return;
    } else if (instruction.compare("stop") == 0
               && inputModeAC == micMode && !buttonIdleBlue) {
        toggleAudioInput();
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
    float confidence = AUDIO_DETECT_THRESHOLD;

    for (int i = 0; i < receivedTensor.size(); i++) {
        float tmp = receivedTensor.at(i);

        if (tmp >= confidence) {
            confidence = tmp;
            label = labelList.at(i);
        }
    }

    uiAC->labelInferenceTimeAC->setText(TEXT_INFERENCE + QString("%1 ms").arg(receivedTimeElapsed));

    if (label != "Unknown") {
        updateDetectedWords(label);
        updateArrow(label);
    }
    qApp->processEvents(QEventLoop::AllEvents);

    if (inputModeAC == micMode) {
        /* If inference failed the first time, run inference a second time but
         * half a second back to ensure the second of data provided to TFL has
         * not completed whilst a word is being spoken */
        if (firstBlock && label == "Unknown") {
            firstBlock = false;
            emit requestInference(content.data() + sampleRate / 2, (size_t) sampleRate * sizeof(float));
        } else {
            firstBlock = true;

            if (secThread.joinable())
                secThread.join();

            secThread = std::thread(&audioCommand::startListening, this);
        }
    } else {
        toggleTalkButtonState();
    }
}

void audioCommand::toggleAudioInput()
{
    if (recordButtonMutex)
        return;

    recordButtonMutex = true;

    toggleTalkButtonState();

    if (!buttonIdleBlue && inputModeAC == micMode) {
        if (setupMic()) {
            if (secThread.joinable())
                secThread.join();

            secThread = std::thread(&audioCommand::startListening, this);
        } else {
            toggleTalkButtonState();
        }
    } else if (buttonIdleBlue && inputModeAC == micMode) {
        if (secThread.joinable())
            secThread.join();

        closeMic();
    } else if (inputModeAC == audioFileMode) {
        startListening();
    }

    recordButtonMutex = false;
}

void audioCommand::readAudioFile(QString filePath)
{
    char* filePath_c = new char [filePath.length() + 1];
    float data[BUFFER_SIZE];
    sf_count_t readFloats;
    SNDFILE* soundFile;
    SF_INFO metaData;

    if(!buttonIdleBlue)
        toggleAudioInput();

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

    inputModeAC = audioFileMode;
    uiAC->actionLoad_Periph->setEnabled(true);
}

bool audioCommand::checkForVolIncrease()
{
    float currentAverage = 0;

    if (content.size() < sampleRate*2)
        return false;

    for (auto ptr = content.begin() + sampleRate; ptr != content.end(); ptr++)
        currentAverage += std::fabs(*ptr);

    currentAverage /= (content.size()/2);

    return currentAverage > ambiantVol;
}

void audioCommand::setMicMode()
{
    inputModeAC = micMode;
}

void audioCommand::startListening()
{
    bool possibleCommand = false;
    bool micAlive = true;

    if (inputModeAC == micMode && !buttonIdleBlue) {
        while (!possibleCommand && micAlive) {
            micAlive = recordSecond();
            possibleCommand = checkForVolIncrease();

            /* Allow the GUI to be updated between read cycles */
            qApp->processEvents(QEventLoop::AllEvents);
        }
    }

    if (inputModeAC == audioFileMode)
        emit requestInference(content.data(), (size_t) sampleRate * sizeof(float));
    else if (possibleCommand && inputModeAC == micMode && !buttonIdleBlue)
        emit requestInference(content.data() + sampleRate, (size_t) sampleRate * sizeof(float));
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

    /* Clear history on keyword "off" */
    if (word.compare("off") == 0) {
        history.clear();
        activeCommands.clear();
        table->setRowCount(0);
    } else {
        history.append(" " + word);
    }

    /* Active commands update */
    uiAC->commandReaderAC->setText(history);
}

bool audioCommand::setupMic()
{
    int err;
    bool retVal = false;

    /* ALSA setup */
    if ((err = snd_pcm_open(&mic_pcm, MIC_DEVICE, SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        qWarning(MIC_OPEN_WARNING);
        emit micWarning(MIC_OPEN_WARNING);
    } else if ((err = snd_pcm_hw_params_malloc(&hw_params)) < 0) {
        qWarning(MIC_HW_PARAM_ALLOC_WARNING);
    } else if ((err = snd_pcm_hw_params_any(mic_pcm, hw_params)) < 0) {
        qWarning(MIC_HW_DETAILS_WARNING);
        emit micWarning(MIC_HW_DETAILS_WARNING);
    } else if ((err = snd_pcm_hw_params_set_access(mic_pcm, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
        qWarning(MIC_ACCESS_TYPE_WARNING);
        emit micWarning(MIC_ACCESS_TYPE_WARNING);
    } else if ((err = snd_pcm_hw_params_set_format(mic_pcm, hw_params, SND_PCM_FORMAT_FLOAT)) < 0) {
        qWarning(MIC_SAMPLE_FORMAT_WARNING);
        emit micWarning(MIC_SAMPLE_FORMAT_WARNING);
    } else if ((err = snd_pcm_hw_params_set_rate(mic_pcm, hw_params, sampleRate, 0)) < 0) {
        qWarning(MIC_SAMPLE_RATE_WARNING);
        emit micWarning(MIC_SAMPLE_RATE_WARNING);
    } else if ((err = snd_pcm_hw_params_set_channels(mic_pcm, hw_params, MIC_CHANNELS)) < 0) {
        qWarning(MIC_CHANNEL_WARNING);
        emit micWarning(MIC_CHANNEL_WARNING);
    } else if ((err = snd_pcm_hw_params(mic_pcm, hw_params)) < 0) {
        qWarning(MIC_PARAMETER_SET_WARNING);
        emit micWarning(MIC_PARAMETER_SET_WARNING);
    } else if ((err = snd_pcm_prepare(mic_pcm)) < 0) {
        qWarning(MIC_PREPARE_WARNING);
        emit micWarning(MIC_PREPARE_WARNING);
    } else {
        /* Take an average of the ambiant background noise volume from the microphone,
         * then add an overhead percentage to that to make a threshold volume to listen for */
        content.clear();

        if (!recordSecond())
            return false;

        for (auto ptr = content.begin(); ptr != content.end(); ptr++)
            ambiantVol += std::fabs(*ptr);

        ambiantVol /= content.size();
        ambiantVol *= BACKGROUND_OVERHEAD;

        content.clear();

        retVal = true;
    }

    return retVal;
}

void audioCommand::closeMic()
{
    /* Close data structures */
    snd_pcm_close(mic_pcm);
    snd_pcm_hw_params_free(hw_params);
}

bool audioCommand::recordSecond()
{
    int err;
    float buffer[sampleRate];

    if (buttonIdleBlue)
        return false;

    qApp->processEvents(QEventLoop::AllEvents);

    /* The last parameter is the number of frames to read from the mic.
     * We match that to our rate per second to retreive 1 second worth of data from the microphone */
    err = snd_pcm_readi(mic_pcm, buffer, sampleRate);

    if (err == -EPIPE) {
        qWarning(READ_OVERRUN_ERROR);
        return false;
    } else if (err == -EBADFD) {
        qWarning(READ_PCM_ERROR);
        return false;
    } else if (err == -ESTRPIPE) {
        qWarning(READ_SUSPEND_ERROR);
        return false;
    } else if (err < 0) {
        qWarning() << "error: Microphone returned:" << err;
        return false;
    } else if ((unsigned int) err != sampleRate) {
        qWarning(READ_INCOMPLETE_WARNING);
        return false;
    }

    /* Clear the second before last */
    if (content.size() > sampleRate)
        content.erase(content.begin(), content.begin() + sampleRate);

    /* Place new elements to the back of the vector */
    content.insert(content.end(), buffer, buffer+sampleRate);

    qApp->processEvents(QEventLoop::AllEvents);

    return true;
}

void audioCommand::toggleTalkButtonState()
{
    buttonIdleBlue = !buttonIdleBlue;

    if (buttonIdleBlue) {
        uiAC->pushButtonTalk->setText("Talk");
        uiAC->pushButtonTalk->setStyleSheet(BUTTON_BLUE);
    } else {
        uiAC->pushButtonTalk->setText("Stop");
        uiAC->pushButtonTalk->setStyleSheet(BUTTON_RED);
    }

    qApp->processEvents(QEventLoop::WaitForMoreEvents);
}

audioCommand::~audioCommand()
{
    if (!buttonIdleBlue && inputModeAC == micMode)
        toggleAudioInput();
}
