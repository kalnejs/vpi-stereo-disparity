
#include "Capture.hpp"

#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define DEBUG 0

#ifdef DEBUG
  #define dbg_user_info(x)   cout << "INFO: " << x << endl
  #define dbg_user_err(x)   cout << "ERR: " << x << endl
#else
  #define dbg_user_info(x)
  #define dbg_user_err(x)
#endif

Capture::Capture(){

}

Capture::~Capture(){
  if(_cap.isOpened()) {
    _cap.release();
  }
}

int Capture::open(int camera_index, int width, int height){
  _cap.open(Capture::_gstreamer_pipeline(camera_index, 1280, 720, 30, width, height), CAP_GSTREAMER);

  if(!_cap.isOpened()) {
      dbg_user_err("Unable to open camera");
      return -1;
  }

  return 0;
}

int Capture::read(Mat &frame){

  if(!_cap.isOpened()) {
    dbg_user_err("camera closed");
    return -1;
  }

 _cap.read(frame);
  if (frame.empty()) {
      cerr << "ERROR! blank frame grabbed\n";
      return -1;
  }

  return 0;
}
