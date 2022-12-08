#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

int main(){
    VideoCapture cap("/Users/zhangzikai/Desktop/traffic_light_mp4/2.mp4");
    Mat frame;
    
    while(cap.isOpened()){
        cap>>frame;
        if(frame.empty())
            break;
        imshow("frame",frame);
//        waitKey(0);
    }
    cout<<"finished"<<endl;
    cap.release();
    return 0;
}
