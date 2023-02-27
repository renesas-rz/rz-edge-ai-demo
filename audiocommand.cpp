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

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

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

#define MIC_CHANNELS 1
#define MIC_DEVICE "plughw:0,0"

#define MODEL_SAMPLE_RATE 44100
#define AUDIO_DETECT_THRESHOLD 0.90

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

#define VOLUME_THRESHOLD_MIN		0.05
#define VOLUME_THRESHOLD_MAX		0.7
#define VOLUME_THRESHOLD_DEFAULT	0.2

audioCommand::audioCommand(Ui::MainWindow *ui, QStringList labelFileList, QString inferenceEngine)
{
    uiAC = ui;
    utilAC = new edgeUtils();
    labelList = labelFileList;
    trail = QVector<QGraphicsLineItem*>();
    QGraphicsScene *scene = uiAC->graphicsView->scene();
    QPen pen = QPen(THEME_LIGHT_GRAY);
    int width = uiAC->graphicsView->width();
    int height = uiAC->graphicsView->height();
    inputModeAC = micMode;
    buttonIdleBlue = true;
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

    current_volume_threshold = VOLUME_THRESHOLD_DEFAULT;
    uiAC->volumeThresholdDial->setMinimum(VOLUME_THRESHOLD_MIN * 100);
    uiAC->volumeThresholdDial->setMaximum(VOLUME_THRESHOLD_MAX * 100);
    uiAC->volumeThresholdDial->setValue(VOLUME_THRESHOLD_DEFAULT * 100);
    connect(uiAC->volumeThresholdDial, SIGNAL(valueChanged(int)), this,
	    SLOT(volumeThresholdDialChanged(int)));

    if (audioMode == no_audio_selection)
	    audioMode = audio;

    debug = false;
    if (audioMode == audioDebug or audioMode == audioRecordDebug or audioMode == audioPlaybackDebug)
        debug = true;

    record = false;
    if (audioMode == audioRecord or audioMode == audioRecordDebug)
        record = true;

    playback = false;
    if (audioMode == audioPlayback or audioMode == audioPlaybackDebug)
        playback = true;

    uiAC->micVolume->setEnabled(false);
    uiAC->micVolumeDial->setEnabled(false);
    if (not playback) {
        if (setupAlsaMixer()) {
            uiAC->micVolumeDial->setMinimum(mic_volume_min);
            uiAC->micVolumeDial->setMaximum(mic_volume_max);
            setMicVolume(mic_volume_max);
            uiAC->micVolumeDial->setValue(mic_volume_max);
            connect(uiAC->micVolumeDial, SIGNAL(valueChanged(int)), this,
                    SLOT(micVolumeDialChanged(int)));
            uiAC->micVolume->setEnabled(true);
            uiAC->micVolumeDial->setEnabled(true);
	}
    }

    recording_fd = -1;

    setupArrow();
}

void audioCommand::setupArrow()
{
    QPen pen = QPen(THEME_RENESAS_BLUE);
    QBrush brush = QBrush(THEME_RENESAS_BLUE, Qt::Dense6Pattern);
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
    QPen pen = QPen(THEME_RENESAS_BLUE);

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
        if (secThread.joinable())
	    secThread.join();
        secThread = std::thread(&audioCommand::startListening, this);
    } else {
        toggleTalkButtonState();
    }
}

void audioCommand::toggleAudioInput()
{
    QPoint gridCentre = QPoint(uiAC->graphicsView->width() / 2, uiAC->graphicsView->height() / 2);
    QTableWidget * table = uiAC->tableWidgetAC;

    if (recordButtonMutex)
        return;

    recordButtonMutex = true;

    toggleTalkButtonState();

    if (!buttonIdleBlue && inputModeAC == micMode) {
        if (setupMic()) {
            if (secThread.joinable())
                secThread.join();

            history.clear();
            uiAC->commandReaderAC->setText(history);
            arrow->setPos(gridCentre);
            arrow->setRotation(ARROW_UP);
            clearTrail();
            activeCommands.clear();
            table->setRowCount(0);
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

void audioCommand::volumeThresholdDialChanged(int value)
{
	current_volume_threshold = (float)value / 100.0;
}


void audioCommand::setMicVolume(long volume)
{
	if (alsa_element) {
		if (volume > mic_volume_max)
			mic_volume_current = mic_volume_max;
		else if (volume < mic_volume_min)
			mic_volume_current = mic_volume_min;
		else
			mic_volume_current = volume;
		snd_mixer_selem_set_capture_volume_all(alsa_element,
						       mic_volume_current);
	}
}

void audioCommand::micVolumeDialChanged(int value)
{
	setMicVolume(value);
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
    // FIXME - there may be no MIC connected, which means this logic will need
    // to be fixed.
    uiAC->actionLoad_Periph->setEnabled(true);

    uiAC->micVolume->setEnabled(false);
    uiAC->micVolumeDial->setEnabled(false);
}

void audioCommand::setMicMode()
{
    inputModeAC = micMode;

    uiAC->micVolume->setEnabled(false);
    uiAC->micVolumeDial->setEnabled(false);
    if (not playback) {
        clearAlsaMixer();
        if (setupAlsaMixer()) {
            uiAC->micVolumeDial->setMinimum(mic_volume_min);
            uiAC->micVolumeDial->setMaximum(mic_volume_max);
            setMicVolume(mic_volume_max);
            uiAC->micVolumeDial->setValue(mic_volume_max);
            connect(uiAC->micVolumeDial, SIGNAL(valueChanged(int)), this,
                    SLOT(micVolumeDialChanged(int)));
            uiAC->micVolume->setEnabled(true);
            uiAC->micVolumeDial->setEnabled(true);
	}
    }
}

bool audioCommand::readSecondFromInputStream(float *inputBuffer) {

    if (playback) {
        ssize_t ret = read(recording_fd, inputBuffer, sizeof(float) * sampleRate);
        sleep(1);

        if (ret == -1) {
            qWarning("ERROR: Can't read from file recording.dat");
            return false;
        } else if (ret == 0) {
            qWarning("ERROR: recording.dat EOF reached");
            return false;
        } else if (ret != (long)(sizeof(float) * sampleRate)) {
            qWarning("FIXME: I don't deal with incomplete reads from file "
                   "recording.dat just yet (and maybe never will!)");
            return false;
        } else {
            return true;
        }
    }
    return recordSecond(inputBuffer);
}

/* Buffering and trimming */

#define next_buffer()	((current_buffer == buffer1) ? buffer2 : buffer1)
#define prev_buffer()	next_buffer()

static void save_wav(float *buffer, int sampling_rate, int number_of_samples) {
	static int index = 1;
	char filename[64];
	SF_INFO sfinfo;

	sprintf(filename, "%04d.wav", index++);
	printf("Creating file %s...\n", filename);

	sfinfo.channels = 1;
	sfinfo.samplerate = sampling_rate;
	sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

	SNDFILE * outfile = sf_open(filename, SFM_WRITE, &sfinfo);
	sf_write_float(outfile, buffer, number_of_samples);
	sf_write_sync(outfile);
	sf_close(outfile);
}

static void debug_to_file(float *input_buffer, float *debug_buffer, int buffer_size,
		   float threshold) {
	static FILE *fp = NULL;
	static int sample = 0;
	float marker;
	int i;

	if (fp == NULL) {
		fp = fopen("debug.txt", "w");
	}

	for (i = 0; i < buffer_size; i++) {
		if (i == 0)
			marker = 1.0;
		if (i == 1)
			marker = -1.0;
		if (i == 2)
			marker = 0.0;
		fprintf(fp, "%d %f %f %f %f\n",
			sample++,
			marker,
			input_buffer[i],
			debug_buffer[i],
			threshold
			);
	}
}

static enum word_location locate_word(int sampling_rate, int *left, int *right,
		float *input_buffer, float *working_buffer, float *debug_buffer,
		int buffers_size, float threshold, bool debug) {
	int i, j, slice, _left, _right;
	int window = sampling_rate / 24;
	float max, absolute;

	_left = -1;
	_right = -1;

	for (i = 0, slice = 0; i < buffers_size; i+=window, slice++) {
		max = 0;
		for (j = i; (j < i + window) && (j < buffers_size); j++) {
			absolute = fabs(input_buffer[j]);
			max = absolute > max ? absolute : max;
		}
		working_buffer[slice] = max;
	}

	// Flatten the edges to prevent artifacts on the edges
	working_buffer[0]         = working_buffer[1];
	working_buffer[slice - 1] = working_buffer[slice - 2];

	for (i = 0; i < slice; i++) {
		if (working_buffer[i] >= threshold) {
			if (_left == -1)
				_left = i;
		} else if (_left >= 0) {
			_right = i;
			break;
		}
	}

	if (debug) {
		for (i = 0; i < slice; i++) {
			for (j = i * window;
			     (j < ((i + 1) * window)) && (j < buffers_size);
			     j++
			    ) {
			     debug_buffer[j] = working_buffer[i];
			}
		}
		debug_to_file(input_buffer, debug_buffer, buffers_size,
			      threshold);
	}

	if (_left == -1)
		return WORD_NOT_FOUND;

	if (_left == 0 && _right == -1)
		return WORD_BOTH_EDGES;

	if (_right == -1) {
		*left = _left * window;
		return WORD_RIGHT_EDGE;
	}

	*right = ((_right + 1) * window) - 1;

	if (_left == 0)
		return WORD_LEFT_EDGE;

	*left = _left * window;

	return WORD_FOUND;
}

void audioCommand::processWordsFromInputStream(int sampling_rate, bool debug) {
	bool run_inference = false;
	int buffer_size = sampling_rate;

	if (!buffer1 or !buffer2 or !buffer3 or !debug_buffer or !working_buffer)
		return;

	while (true) {
		/* If do_we_read is set to false, it means we already read
		 * current_buffer, but we haven't processed it yet, therefore
		 * just process current_buffer without reading a new one.
		 * If do_we_read is set to true, then read a new buffer and
		 * process it.
		 */
		if (do_we_read)
			if (not readSecondFromInputStream(current_buffer))
				return;

		do_we_read = true;

		if (save_after_read) {
			memcpy(
			       buffer3,
			       &(prev_buffer()[sample_left]),
			       sizeof(float)*(buffer_size - sample_left)
			       );
			memcpy(
			       &buffer3[buffer_size - sample_left],
			       current_buffer,
			       sizeof(float)*sample_right
			       );
			if (debug)
				save_wav(buffer3, sampling_rate, buffer_size);
			emit requestInference(buffer3,
				(size_t) sampling_rate * sizeof(float));

			save_after_read = false;
			do_we_read = false;

			return;
		}

		current_search = locate_word(sampling_rate,
					     &current_left, &current_right,
					     current_buffer, working_buffer,
					     debug_buffer, buffer_size,
					     current_volume_threshold, debug);

		// At the moment we are discarding WORD_BOTH_EDGES
		if (current_search == WORD_FOUND) {
			left = current_left;
			right = current_right;
			save = true;
		} else if (current_search == WORD_LEFT_EDGE && previous_search == WORD_RIGHT_EDGE) {
			left = previous_left - buffer_size;
			right = current_right;
			save = true;
		}  else if (current_search == WORD_LEFT_EDGE) {
			if (not first_sample) {
				left = -1;
				right = current_right;
				save = true;
			}
		} else if (previous_search == WORD_RIGHT_EDGE) {
			right = 0;
			left =  previous_left - buffer_size;
			save = true;
		}

		if (save) {
			sample_center = (right + left) / 2;
			sample_left = sample_center - (sampling_rate / 2 );
			sample_right = sample_center + (sampling_rate / 2);

			if (sample_left < 0) {
				sample_left = buffer_size + sample_left;
				memcpy(
				       buffer3,
				       &(prev_buffer()[sample_left]),
				       sizeof(float)*(buffer_size - sample_left)
				       );
				memcpy(
				       &buffer3[buffer_size - sample_left],
				       current_buffer,
				       sizeof(float)*sample_right
				       );
				if (debug)
					save_wav(buffer3, sampling_rate, buffer_size);
				emit requestInference(buffer3,
					(size_t) sampling_rate * sizeof(float));

				run_inference = true;
			} else if (sample_right < buffer_size) {
				if (debug)
					save_wav(current_buffer, sampling_rate, buffer_size);
				emit requestInference(buffer3,
					(size_t) sampling_rate * sizeof(float));
				run_inference = true;
			} else if (sample_right >= buffer_size) {
				// We need more samples to get the word nicely
				// centered
				save_after_read = true;
				sample_right = sample_right - buffer_size;
			}
		}

		first_sample = false;
		previous_search = current_search;
		current_buffer = next_buffer();
		save = false;
		previous_left = current_left;

		if (run_inference)
			break;
	}
}

void audioCommand::startListening()
{
    if (inputModeAC == micMode && !buttonIdleBlue) {
        processWordsFromInputStream(sampleRate, debug);
    } else if (inputModeAC == audioFileMode) {
        emit requestInference(content.data(), (size_t) sampleRate * sizeof(float));
    }
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

    history.append(" " + word);

    /* Active commands update */
    uiAC->commandReaderAC->setText(history);
}

bool audioCommand::setupAlsaMixer()
{
	snd_mixer_elem_t *element = NULL;
	int count, i, err;
	long min, max;

	err = snd_mixer_open(&alsa_handle, 0);
	if (err) {
		qWarning("Warning: I can't open ALSA mixer");
		return false;
	}

	err = snd_mixer_attach(alsa_handle, alsa_card);
	if (err)
		goto alsa_close;

	err = snd_mixer_selem_register(alsa_handle, NULL, NULL);
	if (err)
		goto alsa_detach;

	err = snd_mixer_load(alsa_handle);
	if (err)
		goto alsa_free;

	count = snd_mixer_get_count(alsa_handle);
	element = snd_mixer_first_elem(alsa_handle);
	for (i = 0; i < count; i++) {
		if (snd_mixer_selem_has_capture_volume (element)) {
			alsa_element = element;
			break;
		}
		element = snd_mixer_elem_next(element);
	}

	if (alsa_element == NULL)
		goto alsa_free;

	err = snd_mixer_selem_get_capture_volume_range(alsa_element, &min, &max);
	if (err)
		goto alsa_free;

	mic_volume_min = min;
	mic_volume_max = max;

	return true;

alsa_free:
	snd_mixer_free(alsa_handle);

alsa_detach:
	snd_mixer_detach(alsa_handle, alsa_card);

alsa_close:
	snd_mixer_close(alsa_handle);
	alsa_handle = NULL;
	alsa_element = NULL;

        qWarning("Warning: I can't use ALSA mixer");

	return false;
}

void audioCommand::clearAlsaMixer()
{
	if (alsa_handle == NULL)
		return;

	snd_mixer_free(alsa_handle);
	snd_mixer_detach(alsa_handle, alsa_card);
	snd_mixer_close(alsa_handle);
	alsa_handle = NULL;
	alsa_element = NULL;
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

        // Buffering and trimming buffers
        buffer1 = (float*)malloc(sizeof(float) * sampleRate);
        buffer2 = (float*)malloc(sizeof(float) * sampleRate);
        buffer3 = (float*)malloc(sizeof(float) * sampleRate);
        debug_buffer = (float*)malloc(sizeof(float) * sampleRate);
        working_buffer = (float*)malloc(sizeof(float) * sampleRate);
        left = 0;
        right = 0;
        sample_left = 0;
        sample_right = 0;
        sample_center = 0;
        current_left = 0;
        current_right = 0;
        previous_left = 0;
        current_search = WORD_NOT_FOUND;
        previous_search = WORD_NOT_FOUND;
        current_buffer = buffer1;
        save_after_read = false;
        first_sample = true;
        do_we_read = true;
        save = false;

        if (record)
            recording_fd = open("recording.dat", O_CREAT | O_TRUNC | O_WRONLY,
                                S_IRUSR | S_IWUSR);

        if (playback)
            recording_fd = open("recording.dat", O_RDONLY);

        // content contains the samples to send for inference
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


    free(working_buffer);
    free(debug_buffer);
    free(buffer3);
    free(buffer2);
    free(buffer1);

    buffer1 = NULL;
    buffer2 = NULL;
    buffer3 = NULL;
    debug_buffer = NULL;
    working_buffer = NULL;

    if (recording_fd != -1)
        close(recording_fd);
}

bool audioCommand::recordSecond(float *inputBuffer)
{
    int err;

    if (buttonIdleBlue)
        return false;

    /* The last parameter is the number of frames to read from the mic.
     * We match that to our rate per second to retreive 1 second worth of data from the microphone */
    err = snd_pcm_readi(mic_pcm, inputBuffer, sampleRate);

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
    } else if (record) {
        ssize_t ret = write(recording_fd, inputBuffer, sizeof(float) * sampleRate);

        if (ret == -1) {
            qWarning("ERROR: Can't write to file recording.dat");
        } else if (ret != (long)sizeof(float) * sampleRate) {
            qWarning("FIXME: I don't deal with incomplete writes to "
                   "recording.dat just yet (and maybe never will!)");
        }
    }

    return true;
}

void audioCommand::toggleTalkButtonState()
{
    buttonIdleBlue = !buttonIdleBlue;

    if (buttonIdleBlue) {
        uiAC->pushButtonTalk->setText("Speak\nCommands");
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

    if (not playback)
        clearAlsaMixer();
}
