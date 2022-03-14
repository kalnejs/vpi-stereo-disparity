#include <string>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Capture {

  const int CAMERA_MAX_QUEUE_SIZE = 4;

  public:
    Capture();
    ~Capture();

    int open(int camera_index = 0, int width=1280/2, int height=720/2);
    int read(Mat &frame);

  private:
    VideoCapture _cap;
    static string _gstreamer_pipeline (int id = 0, int capture_width = 1280,
                                      int capture_height = 720, int framerate = 30, int width = 1280, int height = 720) {
        return "nvarguscamerasrc sensor-id="+to_string(id)+
          " ! video/x-raw(memory:NVMM), width=(int)"+to_string(capture_width)+
          ", height=(int)"+to_string(capture_height)+
          ", framerate=(fraction)"+to_string(framerate)+
          "/1 ! nvvidconv flip-method=0"+
          " ! video/x-raw, width=(int)"+to_string(width)+
          ", height=(int)" +to_string(height)+
          ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
      }
};

