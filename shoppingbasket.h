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
#ifndef SHOPPINGBASKET_H
#define SHOPPINGBASKET_H

#include <QMainWindow>
#include <opencv2/videoio.hpp>

#define BUTTON_BLUE "background-color: rgba(42, 40, 157);color: rgb(255, 255, 255);border: 2px;border-radius: 55px;border-style: outset;"
#define BUTTON_GREYED_OUT "background-color: rgba(42, 40, 157, 90);color: rgb(255, 255, 255);border: 2px;border-radius: 55px;border-style: outset;"

#define TEXT_INFERENCE "Inference Time: "
#define TEXT_LOAD_IMAGE "Load Image"
#define TEXT_LOAD_NEW_IMAGE "Load New Image"
#define TEXT_TOTAL_ITEMS "Total Items: "

#define DETECT_THRESHOLD 0.5

class QGraphicsScene;

namespace Ui { class MainWindow; }

enum InputSB { cameraModeSB, imageModeSB };

class shoppingBasket  : public QObject
{
    Q_OBJECT

public:
    shoppingBasket(Ui::MainWindow *ui, QStringList labelFileList, QString pricesFile);
    void setImageMode(bool imageStatus);

public slots:
    void runInference(QVector<float> receivedTensor, int receivedTimeElapsed, const cv::Mat&receivedMat);

signals:
    void getFrame();
    void getStaticImage();
    void getBoxes(const QVector<float>& receivedTensor, QStringList labelList);
    void sendMatToView(const cv::Mat&receivedMat);
    void startVideo();
    void stopVideo();

private slots:
    void processBasket();
    void nextBasket();

private:
    void setNextButton(bool enable);
    void setProcessButton(bool enable);
    std::vector<float> readPricesFile(QString pricesPath);
    QVector<float> sortTensor(QVector<float> &receivedTensor);

    Ui::MainWindow *uiSB;
    QStringList labelListSorted;
    QVector<float> outputTensor;
    std::vector<float> costs;
    QStringList labelList;
    InputSB inputModeSB;
};

#endif // SHOPPINGBASKET_H
