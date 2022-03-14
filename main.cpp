#include "Capture.hpp"

#include <vpi/OpenCVInterop.hpp>

#include <vpi/algo/Remap.h>
#include <vpi/algo/ConvertImageFormat.h>

 #include <vpi/LensDistortionModels.h>

#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>

using namespace std;
using namespace cv;

 #define CHECK_STATUS(STMT)                                    \
     do                                                        \
     {                                                         \
         VPIStatus status = (STMT);                            \
         if (status != VPI_SUCCESS)                            \
         {                                                     \
             char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
             vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
             std::ostringstream ss;                            \
             ss << vpiStatusGetName(status) << ": " << buffer; \
             throw std::runtime_error(ss.str());               \
         }                                                     \
     } while (0);


int main(void){

    int status;

    Capture cL, cR;

    VPIImage vpiFrame1 = NULL, vpiFrame2 = NULL;
    VPIImage vpiConvert1 = NULL, vpiConvert2 = NULL;
    VPIImage vpiRectify1 = NULL, vpiRectify2 = NULL;
    VPIImage vpiOutput1 = NULL, vpiOutput2 = NULL;


    VPIWarpMap vpiWarpMap1 = {},  vpiWarpMap2 = {};
    VPIPolynomialLensDistortionModel vpiDist1 = {}, vpiDist2 = {};
    VPICameraIntrinsic vpiK1 = {}, vpiK2 = {};
    VPICameraExtrinsic vpiX1 = {}, vpiX2 = {};
    VPIPayload vpiRemap1 = NULL, vpiRemap2 = NULL;


    VPIStream vpiStream1 = NULL, vpiStream2 = NULL;


    Mat M1, M2;
    Mat D1, D2;
    Mat P1, P2;
    Mat R1, R2;

    status = cL.open(1);

    if(status){
        return -1;
    }

    status = cR.open(0);

    if(status){
        return -1;
    }

    FileStorage fs("camera_intrinsics.xml", FileStorage::READ);

    if(!fs.isOpened()){
        return -1;
    }

    fs["M1"] >> M1;
    fs["M2"] >> M2;
    fs["D1"] >> D1;
    fs["D2"] >> D2;

    fs.release();

    fs.open("camera_extrinsics.xml", FileStorage::READ);

    if(!fs.isOpened()){
        return -1;
    }

    fs["P1"] >> P1;
    fs["P2"] >> P2;
    fs["R1"] >> R1;
    fs["R2"] >> R2;

    fs.release();

    // Allocate a dense map.
    vpiWarpMap1.grid.numHorizRegions  = 1;
    vpiWarpMap1.grid.numVertRegions   = 1;
    vpiWarpMap1.grid.regionWidth[0]   = 1280/2;
    vpiWarpMap1.grid.regionHeight[0]  = 720/2;
    vpiWarpMap1.grid.horizInterval[0] = 1;
    vpiWarpMap1.grid.vertInterval[0]  = 1;
    CHECK_STATUS(vpiWarpMapAllocData(&vpiWarpMap1));

    CHECK_STATUS(vpiWarpMapGenerateIdentity(&vpiWarpMap1));

    vpiWarpMap2.grid.numHorizRegions  = 1;
    vpiWarpMap2.grid.numVertRegions   = 1;
    vpiWarpMap2.grid.regionWidth[0]   = 1280/2;
    vpiWarpMap2.grid.regionHeight[0]  = 720/2;
    vpiWarpMap2.grid.horizInterval[0] = 1;
    vpiWarpMap2.grid.vertInterval[0]  = 1;
    CHECK_STATUS(vpiWarpMapAllocData(&vpiWarpMap2));

    CHECK_STATUS(vpiWarpMapGenerateIdentity(&vpiWarpMap2));

    vpiDist1.k1 = D1.at<double>(0);
    vpiDist1.k2 = D1.at<double>(1);
    vpiDist1.p1 = D1.at<double>(2);
    vpiDist1.p2 = D1.at<double>(3);
    vpiDist1.k3 = D1.at<double>(4);
    vpiDist1.k4 = D1.at<double>(5);
    vpiDist1.k5 = D1.at<double>(6);
    vpiDist1.k6 = D1.at<double>(7);

    vpiDist2.k1 = D2.at<double>(0);
    vpiDist2.k2 = D2.at<double>(1);
    vpiDist2.p1 = D2.at<double>(2);
    vpiDist2.p2 = D2.at<double>(3);
    vpiDist2.k3 = D2.at<double>(4);
    vpiDist2.k4 = D2.at<double>(5);
    vpiDist2.k5 = D2.at<double>(6);
    vpiDist2.k6 = D2.at<double>(7);

  
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            vpiK1[i][j] = P1.at<double>(i, j);
            vpiK2[i][j] = P2.at<double>(i, j);
        }
    }

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            vpiX1[i][j] = R1.at<double>(i, j);
            vpiX2[i][j] = R2.at<double>(i, j);
        }
    }


    CHECK_STATUS(vpiWarpMapGenerateFromPolynomialLensDistortionModel(vpiK1, vpiX1, vpiK1, &vpiDist1, &vpiWarpMap1));
    CHECK_STATUS(vpiWarpMapGenerateFromPolynomialLensDistortionModel(vpiK2, vpiX2, vpiK2, &vpiDist2, &vpiWarpMap2));

    CHECK_STATUS(vpiCreateRemap(VPI_BACKEND_CUDA, &vpiWarpMap1, &vpiRemap1));
    CHECK_STATUS(vpiCreateRemap(VPI_BACKEND_CUDA, &vpiWarpMap2, &vpiRemap2));

    vpiWarpMapFreeData(&vpiWarpMap1);
    vpiWarpMapFreeData(&vpiWarpMap2);

    CHECK_STATUS(vpiStreamCreate(VPI_BACKEND_CUDA | VPI_BACKEND_VIC, &vpiStream1));
    CHECK_STATUS(vpiStreamCreate(VPI_BACKEND_CUDA | VPI_BACKEND_VIC, &vpiStream2));
 
    CHECK_STATUS(vpiImageCreate(1280/2, 720/2, VPI_IMAGE_FORMAT_NV12_ER, 0, &vpiConvert1));
    CHECK_STATUS(vpiImageCreate(1280/2, 720/2, VPI_IMAGE_FORMAT_NV12_ER, 0, &vpiConvert2));

    CHECK_STATUS(vpiImageCreate(1280/2, 720/2, VPI_IMAGE_FORMAT_NV12_ER, 0, &vpiRectify1));
    CHECK_STATUS(vpiImageCreate(1280/2, 720/2, VPI_IMAGE_FORMAT_NV12_ER, 0, &vpiRectify2));

    CHECK_STATUS(vpiImageCreate(1280/2, 720/2, VPI_IMAGE_FORMAT_U8, 0, &vpiOutput1));
    CHECK_STATUS(vpiImageCreate(1280/2, 720/2, VPI_IMAGE_FORMAT_U8, 0, &vpiOutput2));

    Ptr<StereoBM> sbm = StereoBM::create(64, 15);

    while (true){

        Mat cvFrame1, cvFrame2;

        if(cL.read(cvFrame1) || cR.read(cvFrame2)){
            continue;
        }

        if (vpiFrame1 == NULL){
            CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvFrame1, 0, &vpiFrame1));
        }else{
            CHECK_STATUS(vpiImageSetWrappedOpenCVMat(vpiFrame1, cvFrame1));
        }

        if (vpiFrame2 == NULL){
            CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvFrame2, 0, &vpiFrame2));
        }else{
            CHECK_STATUS(vpiImageSetWrappedOpenCVMat(vpiFrame2, cvFrame2));
        }

        CHECK_STATUS(vpiSubmitConvertImageFormat(vpiStream1, VPI_BACKEND_CUDA, vpiFrame1, vpiConvert1, NULL));
        CHECK_STATUS(vpiSubmitConvertImageFormat(vpiStream2, VPI_BACKEND_CUDA, vpiFrame2, vpiConvert2, NULL));

        CHECK_STATUS(vpiSubmitRemap(vpiStream1, VPI_BACKEND_CUDA, vpiRemap1, vpiConvert1, vpiRectify1, VPI_INTERP_CATMULL_ROM,
                                         VPI_BORDER_ZERO, 0));

        CHECK_STATUS(vpiSubmitRemap(vpiStream2, VPI_BACKEND_CUDA, vpiRemap2, vpiConvert2, vpiRectify2, VPI_INTERP_CATMULL_ROM,
                                         VPI_BORDER_ZERO, 0));
  
        CHECK_STATUS(vpiSubmitConvertImageFormat(vpiStream1, VPI_BACKEND_CUDA, vpiRectify1, vpiOutput1, NULL));
        CHECK_STATUS(vpiSubmitConvertImageFormat(vpiStream2, VPI_BACKEND_CUDA, vpiRectify2, vpiOutput2, NULL));

        CHECK_STATUS(vpiStreamSync(vpiStream1));
        CHECK_STATUS(vpiStreamSync(vpiStream2));

        VPIImageData imgdata1, imgdata2;
        CHECK_STATUS(vpiImageLock(vpiOutput1, VPI_LOCK_READ, &imgdata1));
        CHECK_STATUS(vpiImageLock(vpiOutput2, VPI_LOCK_READ, &imgdata2));

        Mat outFrame1, outFrame2;
        Mat stack;
        CHECK_STATUS(vpiImageDataExportOpenCVMat(imgdata1, &outFrame1));
        CHECK_STATUS(vpiImageDataExportOpenCVMat(imgdata2, &outFrame2));

        for(int i = 0; i < 720/2 / 20; i++){
            line(outFrame1, Point(0,i*20), Point(1280/2,i*20), Scalar(10, 255, 255));
            line(outFrame2, Point(0,i*20), Point(1280/2,i*20), Scalar(10, 255, 255));
        }

        hconcat(outFrame1,outFrame2,stack);

        Mat imgDisparity16S = Mat(1280/2, 720/2, CV_16S);
        Mat imgDisparity8U = Mat(1280/2, 720/2, CV_8UC1);

        sbm->compute(outFrame1, outFrame2, imgDisparity16S);

        double min, max;
        minMaxLoc(imgDisparity16S, &min, &max);

        imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 250/(max - min));

        imshow("disp", imgDisparity8U);
        imshow("stack",stack);
        waitKey(1);
        
        CHECK_STATUS(vpiImageUnlock(vpiOutput1));
        CHECK_STATUS(vpiImageUnlock(vpiOutput2));

        // Mat frame1, frame2;
        // Mat frame1_remap, frame2_remap;
        // Mat stack;
        // int status;

        // status = cam_left.read(frame1);

        // if(status){
        //     continue;
        // }

        // status = cam_right.read(frame2);

        // if(status){
        //     continue;
        // }

        // remap(frame1, frame1_remap, RM1x, RM1y, INTER_LANCZOS4, BORDER_CONSTANT, Scalar(0, 0, 0));
        // remap(frame2, frame2_remap, RM2x, RM2y, INTER_LANCZOS4, BORDER_CONSTANT, Scalar(0, 0, 0));

        // hconcat(frame1_remap, frame2_remap, stack);

        // imshow("img",stack);
        
    }

    return 0;
}