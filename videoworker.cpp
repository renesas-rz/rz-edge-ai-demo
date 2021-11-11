/*****************************************************************************************
 * Copyright (C) 2021 Renesas Electronics Corp.
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
#include <thread>

#include "videoworker.h"

videoWorker::videoWorker(QObject *parent) :
    QObject(parent), stopped(false), running(false), videoDelay(0)
{}

void videoWorker::play_video()
{
    if (!running || stopped) return;

    emit showVideo();

    /* Limit play_video loop speed */
    if (videoDelay != 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(videoDelay));

    /* Use QueuedConnection to give other threads a chance to be processed before next play_video */
    QMetaObject::invokeMethod(this, "play_video", Qt::QueuedConnection);
}

void videoWorker::StopVideo()
{
    stopped = true;
    running = false;
}

void videoWorker::StartVideo()
{
    stopped = false;
    running = true;

    play_video();
}

void videoWorker::setDelayMS(unsigned int delay)
{
    videoDelay = delay;
}
