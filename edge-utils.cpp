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

#include <cmath>

#include "edge-utils.h"

edgeUtils::edgeUtils()
{
    totalProcessTime = 0;
}

void edgeUtils::timeTotalFps(bool startingTimer)
{
    if (startingTimer) {
        /* Start the timer to measure Total FPS */
        startTime = std::chrono::high_resolution_clock::now();
    } else {
        /* Stop timer, calculate Total FPS and display to GUI */
        std::chrono::high_resolution_clock::time_point stopTime = std::chrono::high_resolution_clock::now();
        totalProcessTime = int(std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime).count());
    }
}

float edgeUtils::calculateTotalFps()
{
    float totalFps = 1000.0 / totalProcessTime;

    return  totalFps;
}

float edgeUtils::calculateSigmoid(float realNumber) {
    return (1 / (1 + exp(-realNumber)));
}
