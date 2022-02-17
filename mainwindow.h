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
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/videoio.hpp>

#define TEXT_CAMERA_INIT_STATUS_ERROR "Camera Error!\n\n No camera detected, please check connection and relaunch application.\n\nApplication will now close."
#define TEXT_CAMERA_OPENING_ERROR "Camera Error!\n\n Camera not Opening, please check connection and relaunch application.\n\nApplication will now close."
#define TEXT_CAMERA_FAILURE_ERROR "Camera Error!\n\n Camera has stopped working, please check the connection and relaunch application.\n\nApplication will now close."

#define IMAGE_FILE_FILTER "Images (*.bmp *.dib *.jpeg *.jpg *.jpe *.png *.pbm *.pgm *.ppm *.sr *.ras *.tiff *.tif);;"
#define VIDEO_FILE_FILTER "Videos (*.asf *.avi *.3gp *.mp4 *m4v *.mov *.flv *.mpeg *.mkv *.webm *.mxf *.ogg)"

#define LABEL_DIRECTORY_PATH "/opt/rz-edge-ai-demo/labels/"
#define LABEL_DIRECTORY_OD "/opt/rz-edge-ai-demo/labels/mobilenet_ssd_v2_coco_quant_postprocess_labels.txt"
#define MEDIA_DIRECTORY_PATH "/opt/rz-edge-ai-demo/media/"
#define MODEL_DIRECTORY_PATH "/opt/rz-edge-ai-demo/models/"
#define MODEL_DIRECTORY_OD "/opt/rz-edge-ai-demo/models/mobilenet_ssd_v2_coco_quant_postprocess.tflite"
#define MODEL_DIRECTORY_SB "/opt/rz-edge-ai-demo/models/shoppingBasketDemo.tflite"
#define RENESAS_RZ_LOGO_DIRECTORY "/opt/rz-edge-ai-demo/logos/renesas-rz-logo.png"
#define SPLASH_SCREEN_DIRECTORY "/opt/rz-edge-ai-demo/logos/rz-splashscreen.png"

#define G2E_HW_INFO "Hardware Information\n\nBoard: RZ/G2E ek874\nCPUs: 2x Arm Cortex-A53,\nDDR: 2GB"
#define G2L_HW_INFO "Hardware Information\n\nBoard: RZ/G2L smarc-rzg2l-evk\nCPUs: 2x Arm Cortex-A55\nDDR: 2GB"
#define G2LC_HW_INFO "Hardware Information\n\nBoard: RZ/G2LC smarc-rzg2lc-evk\nCPUs: 2x Arm Cortex-A55\nDDR: 1GB"
#define G2M_HW_INFO "Hardware Information\n\nBoard: RZ/G2M hihope-rzg2m\nCPUs: 2x Arm Cortex-A57, 4x Arm Cortex-A53\nDDR: 4GB"
#define HW_INFO_WARNING "Unknown Board!"

#define APP_WIDTH 1275
#define APP_HEIGHT 635
#define TABLE_COLUMN_WIDTH 180
#define GRAPHICS_VIEW_EXCESS_SPACE 2 //Prevents the image from scaling outside the graphics view
#define BOX_WIDTH 2
#define BOX_COLOUR Qt::green
#define TEXT_COLOUR Qt::green
#define MIPI_VIDEO_DELAY 50

/* Application exit codes */
#define EXIT_OKAY 0
#define EXIT_CAMERA_INIT_ERROR 1
#define EXIT_CAMERA_STOPPED_ERROR 2

class QGraphicsScene;
class QGraphicsView;
class objectDetection;
class opencvWorker;
class shoppingBasket;
class tfliteWorker;
class QElapsedTimer;
class QEventLoop;
class videoWorker;

namespace Ui { class MainWindow; } //Needed for mainwindow.ui

enum Input { cameraMode, imageMode, videoMode };
enum Mode { SB, OD };

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent, QString cameraLocation, QString labelLocation, QString modelLocation);

public slots:
    void ShowVideo();
    void processFrame();

signals:
    void fileLoaded();
    void stopInference();
    void sendMatToDraw(const cv::Mat& matToSend);

private slots:
    void closeEvent(QCloseEvent *event);
    void drawBoxes(const QVector<float>& outputTensor, QStringList labelList);
    void drawMatToView(const cv::Mat& matInput);
    void getImageFrame();
    void on_actionLicense_triggered();
    void on_actionEnable_ArmNN_Delegate_triggered();
    void on_actionShopping_Basket_triggered();
    void on_actionObject_Detection_triggered();
    void on_actionHardware_triggered();
    void on_actionExit_triggered();
    void on_actionAuto_White_Balance_triggered();
    void on_actionAuto_Exposure_triggered();
    void on_actionAuto_Gain_triggered();
    void on_actionLoad_Camera_triggered();
    void on_actionLoad_File_triggered();
    void on_actionLoad_Model_triggered();

private:
    void createTfWorker();
    QImage matToQImage(const cv::Mat& matToConvert);
    void createVideoWorker();
    void errorPopup(QString errorMessage, int errorCode);
    void setupObjectDetectMode();
    void setupShoppingMode();
    void disconnectSignals();
    void checkInputMode();

    Ui::MainWindow *ui;
    unsigned int iterations;
    bool useArmNNDelegate;
    QFont font;
    QPixmap image;
    QGraphicsScene *scene;
    QGraphicsView *graphicsView;
    opencvWorker *cvWorker;
    shoppingBasket *shoppingBasketMode;
    objectDetection *objectDetectMode;
    tfliteWorker *tfWorker;
    QEventLoop *qeventLoop;
    QString boardInfo;
    QString modelPath;
    QString modelObjectDetect;
    QString labelPath;
    videoWorker *vidWorker;
    Input inputMode;
    Mode demoMode;
};

#endif // MAINWINDOW_H
