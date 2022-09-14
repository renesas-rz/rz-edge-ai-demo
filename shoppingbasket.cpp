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

#include "shoppingbasket.h"
#include "ui_mainwindow.h"

#include <QFile>

#define ITEM_INDEX 4
#define BOX_POINTS 4

#define ITEM_COL 0
#define QUANT_COL 1
#define PRICE_COL 2

shoppingBasket::shoppingBasket(Ui::MainWindow *ui, QStringList labelFileList, QString pricesFile,
                               QString modelPath, QString inferenceEngine, bool cameraConnect)
{
    QFont font;
    QString modelName;

    uiSB = ui;
    inputModeSB = cameraMode;
    labelList = labelFileList;
    costs = readPricesFile(pricesFile);
    camConnect = cameraConnect;

    font.setPointSize(EDGE_FONT_SIZE);
    modelName = modelPath.section('/', -1);

    uiSB->actionShopping_Basket->setDisabled(true);
    uiSB->actionObject_Detection->setDisabled(false);
    uiSB->actionPose_Estimation->setDisabled(false);
    uiSB->actionFace_Detection->setDisabled(false);
    uiSB->actionLoad_Camera->setDisabled(true);
    uiSB->actionLoad_File->setText(TEXT_LOAD_IMAGE);

    uiSB->labelAIModelFilenameSB->setText(modelName);
    uiSB->labelInferenceEngineSB->setText(inferenceEngine);
    uiSB->labelInferenceTimeSB->setText(TEXT_INFERENCE);
    uiSB->labelDemoMode->setText("Mode: Shopping Basket");
    uiSB->labelTotalItems->setText(TEXT_TOTAL_ITEMS);

    uiSB->stackedWidgetLeft->setCurrentIndex(STACK_WIDGET_INDEX_SB);
    uiSB->stackedWidgetRight->setCurrentIndex(STACK_WIDGET_INDEX_SB);

    uiSB->tableWidget->verticalHeader()->setDefaultSectionSize(25);
    uiSB->tableWidget->setHorizontalHeaderLabels({"Item", "Quantity", "Unit Price"});
    uiSB->tableWidget->horizontalHeader()->setFont(font);
    uiSB->tableWidget->setEditTriggers(QAbstractItemView::NoEditTriggers);
    uiSB->tableWidget->resizeColumnsToContents();
    double column1Width = uiSB->tableWidget->geometry().width() * 0.5;
    uiSB->tableWidget->setColumnWidth(ITEM_COL, column1Width);
    uiSB->tableWidget->setColumnWidth(QUANT_COL, column1Width * 0.5);
    uiSB->tableWidget->horizontalHeader()->setStretchLastSection(true);
    uiSB->tableWidget->setRowCount(0);

    setProcessButton(true);
    setNextButton(false);
}

std::vector<float> shoppingBasket::readPricesFile(QString pricesPath)
{
    QFile pricesFile;
    QString fileLine;
    std::vector<float> prices;

    pricesFile.setFileName(pricesPath);
    if (!pricesFile.open(QIODevice::ReadOnly | QIODevice::Text))
        qFatal("%s could not be opened.", pricesPath.toStdString().c_str());

    currency = pricesFile.readLine();
    currency.remove(QRegularExpression("\n"));

    while (!pricesFile.atEnd()) {
        fileLine = pricesFile.readLine();
        fileLine.remove(QRegularExpression("\n"));
        prices.push_back(fileLine.toFloat());
    }

    return prices;
}

void shoppingBasket::setNextButton(bool enable)
{
    if (enable) {
        uiSB->pushButtonNextBasket->setStyleSheet(BUTTON_BLUE);
        uiSB->pushButtonNextBasket->setEnabled(true);
    } else {
        uiSB->pushButtonNextBasket->setStyleSheet(BUTTON_GREYED_OUT);
        uiSB->pushButtonNextBasket->setEnabled(false);
    }

    qApp->processEvents(QEventLoop::WaitForMoreEvents);
}

void shoppingBasket::setProcessButton(bool enable)
{
    if (enable) {
           uiSB->pushButtonProcessBasket->setStyleSheet(BUTTON_BLUE);
           uiSB->pushButtonProcessBasket->setEnabled(true);
       } else {
           uiSB->pushButtonProcessBasket->setStyleSheet(BUTTON_GREYED_OUT);
           uiSB->pushButtonProcessBasket->setEnabled(false);
       }

       qApp->processEvents(QEventLoop::WaitForMoreEvents);
}

void shoppingBasket::nextBasket()
{
    setProcessButton(true);
    setNextButton(false);

    uiSB->tableWidget->setRowCount(0);
    uiSB->labelInferenceTimeSB->setText(TEXT_INFERENCE);
    uiSB->labelTotalItems->setText(TEXT_TOTAL_ITEMS);

    if (inputModeSB == imageMode)
        emit getStaticImage();
    else
        emit startVideo();
}

void shoppingBasket::processBasket()
{
    if (inputModeSB == cameraMode)
        emit stopVideo();

    setProcessButton(false);
    setNextButton(true);

    uiSB->tableWidget->setRowCount(0);
    uiSB->labelInferenceTimeSB->setText(TEXT_INFERENCE);

    outputTensor.clear();

    emit getFrame();
}

QVector<float> shoppingBasket::sortTensor(QVector<float> &receivedTensor, int receivedStride)
{
    QVector<float> sortedTensor = QVector<float>();

    /* The final output tensor of the model is unused in this demo mode */
    receivedTensor.removeLast();

    for(int i = receivedStride; i > 0; i--) {
        float confidenceLevel = receivedTensor.at(receivedTensor.size() - i);

        /* Only include the item if the confidence level is at threshold */
        if (confidenceLevel > DETECT_THRESHOLD && confidenceLevel <= float(1.0)) {
            /* Box points */
            for(int j = 0; j < BOX_POINTS; j++)
                sortedTensor.push_back(receivedTensor.at((receivedStride - i) * BOX_POINTS + j));

            /* Item ID */
            sortedTensor.push_back(receivedTensor.at(receivedTensor.size() - (receivedStride * 2) + (receivedStride - i)));

            /* Confidence level */
            sortedTensor.push_back(confidenceLevel);
        }
    }

    return sortedTensor;
}

void shoppingBasket::runInference(QVector<float> receivedTensor, int receivedStride, int receivedTimeElapsed, const cv::Mat &receivedMat)
{
    QTableWidgetItem* item;
    QTableWidgetItem* price;
    QStringList *labelSet = new QStringList();
    float totalCost = 0;
    outputTensor = sortTensor(receivedTensor, receivedStride);

    uiSB->tableWidget->setRowCount(0);
    labelListSorted.clear();

    for (int i = ITEM_INDEX; (i + 1) < outputTensor.size(); i += 6) {
        totalCost += costs[int(outputTensor[i])];
        labelListSorted.push_back(labelList[int(outputTensor[i])]);
    }

    labelListSorted.sort();

    for (int i = 0; i < labelListSorted.size(); i++) {
        int quantityCount = labelListSorted.count(labelListSorted.at(i));
        QTableWidgetItem* item = new QTableWidgetItem(labelListSorted.at(i));
        QTableWidgetItem* quantity = new QTableWidgetItem(QString::number(quantityCount));

        /* Ensure the item is not already displayed in the table */
        if (labelSet->contains(labelListSorted.at(i)))
            continue;

        item->setTextAlignment(Qt::AlignCenter);
        quantity->setTextAlignment(Qt::AlignCenter);

        uiSB->tableWidget->insertRow(uiSB->tableWidget->rowCount());
        uiSB->tableWidget->setItem(uiSB->tableWidget->rowCount()-1, ITEM_COL, item);
        uiSB->tableWidget->setItem(uiSB->tableWidget->rowCount()-1, QUANT_COL, quantity);
        uiSB->tableWidget->setItem(uiSB->tableWidget->rowCount()-1, PRICE_COL,
        price = new QTableWidgetItem(currency + QString::number(
                double(costs[labelList.indexOf(labelListSorted.at(i))]), 'f', 2)));
        price->setTextAlignment(Qt::AlignRight);

        labelSet->push_back(labelListSorted.at(i));
    }

    delete labelSet;

    uiSB->labelInferenceTimeSB->setText(TEXT_INFERENCE + QString("%1 ms").arg(receivedTimeElapsed));
    uiSB->tableWidget->insertRow(uiSB->tableWidget->rowCount());

    item = new QTableWidgetItem("Total Cost:");
    item->setTextAlignment(Qt::AlignBottom | Qt::AlignRight);
    uiSB->tableWidget->setItem(uiSB->tableWidget->rowCount()-1, QUANT_COL, item);

    item = new QTableWidgetItem(currency + QString::number(double(totalCost), 'f', 2));
    item->setTextAlignment(Qt::AlignBottom | Qt::AlignRight);
    uiSB->tableWidget->setItem(uiSB->tableWidget->rowCount()-1, PRICE_COL, item);

    if(!uiSB->pushButtonProcessBasket->isEnabled())
        emit sendMatToView(receivedMat);

    emit getBoxes(outputTensor, labelList);
}

void shoppingBasket::setImageMode(bool imageStatus)
{
    if (imageStatus) {
        inputModeSB = imageMode;
        uiSB->actionLoad_File->setText(TEXT_LOAD_NEW_IMAGE);

        emit getStaticImage();
    } else {
        inputModeSB = cameraMode;
        uiSB->actionLoad_File->setText(TEXT_LOAD_IMAGE);

        emit startVideo();
    }
    uiSB->actionLoad_Camera->setEnabled(imageStatus && camConnect);
    uiSB->labelInferenceTimeSB->setText(TEXT_INFERENCE);
    uiSB->labelTotalItems->setText(TEXT_TOTAL_ITEMS);
    uiSB->tableWidget->setRowCount(0);
    setProcessButton(true);
    setNextButton(false);
}
