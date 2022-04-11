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

const QStringList shoppingBasket::labelList = {"Baked Beans", "Coke", "Diet Coke",
                                               "Fusilli Pasta", "Lindt Chocolate",
                                               "Mars", "Penne Pasta", "Pringles",
                                               "Redbull", "Sweetcorn"};

shoppingBasket::shoppingBasket(Ui::MainWindow *ui, QString pricesFile)
{
    QFont font;
    uiSB = ui;
    inputModeSB = cameraModeSB;
    costs = readPricesFile(pricesFile);

    font.setPointSize(14);

    uiSB->actionShopping_Basket->setDisabled(true);
    uiSB->actionObject_Detection->setDisabled(false);
    uiSB->actionLoad_Camera->setDisabled(true);
    uiSB->actionLoad_File->setText(TEXT_LOAD_IMAGE);

    uiSB->labelInference->setText(TEXT_INFERENCE);
    uiSB->labelDemoMode->setText("Mode: Shopping Basket");
    uiSB->labelTotalItems->setText(TEXT_TOTAL_ITEMS);

    uiSB->stackedWidgetLeft->setCurrentIndex(0);
    uiSB->stackedWidgetRight->setCurrentIndex(0);

    uiSB->tableWidget->verticalHeader()->setDefaultSectionSize(25);
    uiSB->tableWidget->setHorizontalHeaderLabels({"Item", "Price"});
    uiSB->tableWidget->horizontalHeader()->setFont(font);
    uiSB->tableWidget->setEditTriggers(QAbstractItemView::NoEditTriggers);
    uiSB->tableWidget->resizeColumnsToContents();
    double column1Width = uiSB->tableWidget->geometry().width() * 0.8;
    uiSB->tableWidget->setColumnWidth(0, column1Width);
    uiSB->tableWidget->horizontalHeader()->setStretchLastSection(true);
    uiSB->tableWidgetOD->setRowCount(0);

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

    while (!pricesFile.atEnd()) {
        fileLine = pricesFile.readLine();
        fileLine.remove(QRegularExpression("^\\s*\\d*\\s*"));
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
    uiSB->labelInference->setText(TEXT_INFERENCE);
    uiSB->labelTotalItems->setText(TEXT_TOTAL_ITEMS);

    if (inputModeSB == imageModeSB)
        emit getStaticImage();
    else
        emit startVideo();
}

void shoppingBasket::processBasket()
{
    if (inputModeSB == cameraModeSB)
        emit stopVideo();

    setProcessButton(false);
    setNextButton(true);

    uiSB->tableWidget->setRowCount(0);
    uiSB->labelInference->setText(TEXT_INFERENCE);

    outputTensor.clear();

    emit getFrame();
}

void shoppingBasket::runInference(const QVector<float> &receivedTensor, int receivedTimeElapsed, const cv::Mat &receivedMat)
{
    QTableWidgetItem* item;
    QTableWidgetItem* price;
    float totalCost = 0;

    outputTensor = receivedTensor;
    uiSB->tableWidget->setRowCount(0);
    labelListSorted.clear();

    for (int i = 0; (i + 5) < receivedTensor.size(); i += 6) {
        totalCost += costs[int(outputTensor[i])];
        labelListSorted.push_back(labelList[int(outputTensor[i])]);
    }

    labelListSorted.sort();

    for (int i = 0; i < labelListSorted.size(); i++) {
        QTableWidgetItem* item = new QTableWidgetItem(labelListSorted.at(i));
        item->setTextAlignment(Qt::AlignCenter);

        uiSB->tableWidget->insertRow(uiSB->tableWidget->rowCount());
        uiSB->tableWidget->setItem(uiSB->tableWidget->rowCount()-1, 0, item);
        uiSB->tableWidget->setItem(uiSB->tableWidget->rowCount()-1, 1,
        price = new QTableWidgetItem("£" + QString::number(
                double(costs[labelList.indexOf(labelListSorted.at(i))]), 'f', 2)));
        price->setTextAlignment(Qt::AlignRight);
    }

    uiSB->labelInference->setText(TEXT_INFERENCE + QString("%1 ms").arg(receivedTimeElapsed));
    uiSB->tableWidget->insertRow(uiSB->tableWidget->rowCount());

    item = new QTableWidgetItem("Total Cost:");
    item->setTextAlignment(Qt::AlignBottom | Qt::AlignRight);
    uiSB->tableWidget->setItem(uiSB->tableWidget->rowCount()-1, 0, item);

    item = new QTableWidgetItem("£" + QString::number(double(totalCost), 'f', 2));
    item->setTextAlignment(Qt::AlignBottom | Qt::AlignRight);
    uiSB->tableWidget->setItem(uiSB->tableWidget->rowCount()-1, 1, item);

    if(!uiSB->pushButtonProcessBasket->isEnabled())
        emit sendMatToView(receivedMat);

    emit getBoxes(outputTensor, labelList);
}

void shoppingBasket::setImageMode(bool imageStatus)
{
    if (imageStatus) {
        inputModeSB = imageModeSB;
        uiSB->actionLoad_File->setText(TEXT_LOAD_NEW_IMAGE);

        emit getStaticImage();
    } else {
        inputModeSB = cameraModeSB;
        uiSB->actionLoad_File->setText(TEXT_LOAD_IMAGE);
    }
    uiSB->actionLoad_Camera->setEnabled(imageStatus);
    uiSB->labelInference->setText(TEXT_INFERENCE);
    uiSB->labelTotalItems->setText(TEXT_TOTAL_ITEMS);
    uiSB->tableWidget->setRowCount(0);
    setProcessButton(true);
    setNextButton(false);
}
