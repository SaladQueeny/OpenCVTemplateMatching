#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2\tracking.hpp"
#include <math.h>
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>
using namespace cv;
using namespace std;

vector<Mat> image_array;
Mat quad;
const int FRAME_WIDTH = 1280;
const int FRAME_HEIGHT = 720;
vector<Point2f> good_points;
bool check = false;

string intToString(int number) {
    stringstream ss;
    ss << number;
    return ss.str();
}
void match(Mat ref, Mat tpl)
{
    good_points.clear();
    check = false;

    Mat gref, gtpl;
    cvtColor(ref, gref, COLOR_BGR2GRAY);
    cvtColor(tpl, gtpl, COLOR_BGR2GRAY);

    const int low_canny = 110;
    Canny(gref, gref, low_canny, low_canny * 3);
    Canny(gtpl, gtpl, low_canny, low_canny * 3);

    imshow("file", gref);
    imshow("template", gtpl);

    Mat res_32f(ref.rows - tpl.rows + 1, ref.cols - tpl.cols + 1, CV_32FC1);
    matchTemplate(gref, gtpl, res_32f, TM_CCOEFF_NORMED);

    Mat res;
    res_32f.convertTo(res, CV_8U, 255.0);
    imshow("result", res);

    int size = ((tpl.cols + tpl.rows) / 4) * 2 + 1;
    adaptiveThreshold(res, res, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, size, -64);
    imshow("result_thresh", res);
    int count = 0;
    vector<Point> points;
    while (1)
    {
        double minval, maxval;
        Point minloc, maxloc;
        minMaxLoc(res, &minval, &maxval, &minloc, &maxloc);
        if (maxval > 0)
        {
            count++;
            rectangle(ref, maxloc, Point(maxloc.x + tpl.cols, maxloc.y + tpl.rows), Scalar(0, 255, 0), 2);
            putText(ref, "[" + intToString(maxloc.x) + " " + intToString(maxloc.y) + "]", maxloc, 1, 1, Scalar(0, 255, 0), 2);
            floodFill(res, maxloc, 0);
            cout << maxloc << endl;
            points.push_back(maxloc);
        }
        else
            break;
        cout << count << endl;
    }
    cout << endl;
    if (points.size() >= 4) {
        Point prev = points[0];
        good_points.push_back(prev);
        for (Point i : points) {
            if ((abs(i.x - prev.x) > 5) || (abs(i.y - prev.y) > 5)) {
                good_points.push_back(i);
            }
            prev = i;
        }
        cout << endl;
        for (Point i : good_points) {
            cout << i << endl;
        }
        check = true;
    }


    imshow("final", ref);
    //waitKey(0);
    //destroyAllWindows();
    //return 0;
}
void find_triangles(Mat image) {
    good_points.clear();
    check = false;
    Mat img_grey;
    cvtColor(image, img_grey, COLOR_BGR2GRAY);
    //Mat bin;
    //inRange(img_grey, Scalar(40), Scalar(150), bin);
    Mat image_blurred;
    GaussianBlur(img_grey, image_blurred, Size(3, 3), 0);
    //imshow("Image Blurred with Gaussian Kernel", image_blurred);

    Mat img_cny;
    //img_cny = image_blurred;
    const int low_canny = 100;
    Canny(image_blurred, img_cny, 100, 300);
    //imshow("Image after canny", img_cny);

    Mat img_dil;
    //img_dil = img_cny;
    dilate(img_cny, img_dil, getStructuringElement(MORPH_RECT, Size(5, 5)));
    //dilate(img_dil, img_dil, getStructuringElement(MORPH_RECT, Size(5, 5)));
    //imshow("After dilation", img_dil);

    Mat img_erode;
    erode(img_dil, img_erode, getStructuringElement(MORPH_RECT, Size(3, 3)));
    //img_erode = img_dil;
    //imshow("After erode", img_erode);

    //RNG rng(12345);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(img_erode, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat drawing = Mat::zeros(img_erode.size(), CV_8UC3);
    cout << img_erode.size() << endl;
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(drawing, contours, (int)i, Scalar(0, 255, 0), 2, LINE_8, hierarchy, 0);
    }
    vector<Point> approxTriangle;
    const int MIN_OBJECT_AREA = 12 * 12;
    const int MAX_OBJECT_AREA = 20 * 20;
    if (hierarchy.size() > 0) {
        int numObjects = hierarchy.size();

        for (int index = 0; index < contours.size(); index++) {
            Moments moment = moments((Mat)contours[index]);
            double area = moment.m00;
            double x = moment.m10 / area;
            double y = moment.m01 / area;
            int size = arcLength(contours[index], true);
            //cout << "size " << size << " ";
            //cout << "area " << area << " ";
            approxPolyDP(contours[index], approxTriangle, arcLength(Mat(contours[index]), true) * 0.2, true);
            //cout << approxTriangle.size() << endl;
            //if (arcLength(contours[index], true) < 350 && arcLength(contours[index], true) > 200 && approxTriangle.size() == 3) {
            if (approxTriangle.size() == 3 && area > 1000) {
             
                cout << "Find good" << endl;
                cout << "x:" << x << " y:" << y << " " << area << endl;
                cout << "x:" << contours[index][2].x << " y:" << contours[index][2].y << " " << area << endl;

                good_points.push_back(Point(x, y));
                drawContours(image, contours, index, Scalar(255, 255, 0), 2, LINE_8, hierarchy, 0);
                drawContours(drawing, contours, index, Scalar(255, 255, 0), 2, LINE_8, hierarchy, 0);
                putText(drawing, intToString(area), Point(contours[index][2].x, contours[index][2].y), 2, 1, Scalar(0, 255, 0), 2);
            }
        }
    }
    if (good_points.size() == 4) {
        check = true;
    }
    imshow("Contours", drawing);
}

void sortCorners(std::vector<cv::Point2f>& corners, Point2f center)
{
    std::vector<cv::Point2f> top, bot;
    for (int i = 0; i < corners.size(); i++)
    {
        if (corners[i].y < center.y)
            top.push_back(corners[i]);
        else
            bot.push_back(corners[i]);
    }
    corners.clear();
    if (top.size() == 2 && bot.size() == 2) {
        cv::Point2f tl = top[0].x > top[1].x ? top[1] : top[0];
        cv::Point2f tr = top[0].x > top[1].x ? top[0] : top[1];
        cv::Point2f bl = bot[0].x > bot[1].x ? bot[1] : bot[0];
        cv::Point2f br = bot[0].x > bot[1].x ? bot[0] : bot[1];
        corners.push_back(tl);
        corners.push_back(tr);
        corners.push_back(br);
        corners.push_back(bl);
    }
}
void Homography_for_matchTemplate(Mat src, Mat templ) {
    Point2f center(0, 0);
    Mat bw;
    vector<cv::Point2f> corners;
    corners.push_back(Point(good_points[0].x + templ.size().width / 2, good_points[0].y + templ.size().height / 2));
    corners.push_back(Point(good_points[1].x + templ.size().width / 2, good_points[1].y + templ.size().height / 2));
    corners.push_back(Point(good_points[2].x + templ.size().width / 2, good_points[2].y + templ.size().height / 2));
    corners.push_back(Point(good_points[3].x + templ.size().width / 2, good_points[3].y + templ.size().height / 2));
    vector<cv::Point2f> approx;
    approx.push_back(Point(good_points[0].x + templ.size().width / 2, good_points[0].y + templ.size().height / 2));
    approx.push_back(Point(good_points[1].x + templ.size().width / 2, good_points[1].y + templ.size().height / 2));
    approx.push_back(Point(good_points[2].x + templ.size().width / 2, good_points[2].y + templ.size().height / 2));
    approx.push_back(Point(good_points[3].x + templ.size().width / 2, good_points[3].y + templ.size().height / 2));
    cout << approx.size() << endl;
    if (approx.size() != 4)
    {
        cout << "The object is not quadrilateral!" << endl;
        return;
    }
    // Get mass center
    for (int i = 0; i < corners.size(); i++)
        center += corners[i];
    center *= (1. / corners.size());
    sortCorners(corners, center);
    if (corners.size() == 0) {
        cout << "The corners were not sorted correctly!" << endl;
        return;
    }
    Mat dst = src.clone();
    // Draw corner points
    circle(dst, Point(good_points[0].x + templ.size().width / 2, good_points[0].y + templ.size().height / 2), 3, CV_RGB(255, 0, 0), 2);
    circle(dst, Point(good_points[1].x + templ.size().width / 2, good_points[1].y + templ.size().height / 2), 3, CV_RGB(0, 255, 0), 2);
    circle(dst, Point(good_points[2].x + templ.size().width / 2, good_points[2].y + templ.size().height / 2), 3, CV_RGB(0, 0, 255), 2);
    circle(dst, Point(good_points[3].x + templ.size().width / 2, good_points[3].y + templ.size().height / 2), 3, CV_RGB(255, 255, 255), 2);
    // Draw mass center
    circle(dst, center, 3, CV_RGB(255, 255, 0), 2);
    //Mat quad = cv::Mat::zeros(300, 220, CV_8UC3);
    quad = cv::Mat::zeros(300, 220, CV_8UC3);
    vector<cv::Point2f> quad_pts;
    quad_pts.push_back(cv::Point2f(0, 0));
    quad_pts.push_back(cv::Point2f(quad.cols, 0));
    quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));
    quad_pts.push_back(cv::Point2f(0, quad.rows));
    Mat transmtx = getPerspectiveTransform(corners, quad_pts);
    warpPerspective(src, quad, transmtx, quad.size());
    imshow("image", dst);
    imshow("povernutoe", quad);
    corners.clear();
    approx.clear();
}
void Homography(Mat src) {
    Point2f center(0, 0);
    Mat bw;
    vector<cv::Point2f> corners;
    corners.push_back(Point(good_points[0].x, good_points[0].y));
    corners.push_back(Point(good_points[1].x, good_points[1].y));
    corners.push_back(Point(good_points[2].x, good_points[2].y));
    corners.push_back(Point(good_points[3].x, good_points[3].y));
    vector<cv::Point2f> approx;
    approx.push_back(Point(good_points[0].x, good_points[0].y));
    approx.push_back(Point(good_points[1].x, good_points[1].y));
    approx.push_back(Point(good_points[2].x, good_points[2].y));
    approx.push_back(Point(good_points[3].x, good_points[3].y));
    cout << approx.size() << endl;
    if (approx.size() != 4)
    {
        cout << "The object is not quadrilateral!" << std::endl;
        return;
    }
    // Get mass center
    for (int i = 0; i < corners.size(); i++)
        center += corners[i];
    center *= (1. / corners.size());
    sortCorners(corners, center);
    if (corners.size() == 0) {
        cout << "The corners were not sorted correctly!" << std::endl;
        return;
    }
    Mat dst = src.clone();
    // Draw corner points
    circle(dst, Point(good_points[0].x, good_points[0].y), 3, CV_RGB(255, 0, 0), 2);
    circle(dst, Point(good_points[1].x, good_points[1].y), 3, CV_RGB(0, 255, 0), 2);
    circle(dst, Point(good_points[2].x, good_points[2].y), 3, CV_RGB(0, 0, 255), 2);
    circle(dst, Point(good_points[3].x, good_points[3].y), 3, CV_RGB(255, 255, 255), 2);
    // Draw mass center
    circle(dst, center, 3, CV_RGB(255, 255, 0), 2);
    //Mat quad = cv::Mat::zeros(300, 220, CV_8UC3);
    vector<int> ppp(4);
    ppp[0] = sqrt(pow(corners[0].x - corners[1].x, 2) + pow(corners[0].y - corners[1].y, 2));
    ppp[1] = sqrt(pow(corners[2].x - corners[1].x, 2) + pow(corners[2].y - corners[1].y, 2));
    ppp[2] = sqrt(pow(corners[2].x - corners[3].x, 2) + pow(corners[2].y - corners[3].y, 2));
    ppp[3] = sqrt(pow(corners[3].x - corners[0].x, 2) + pow(corners[3].y - corners[0].y, 2));
    sort(ppp.begin(), ppp.end());
    cout << endl;
    cout << "Begin of good_points" << endl;
    for (Point i : corners) {
        cout << i << endl;
    }
    cout << "end of good_points" << endl;
    cout << endl;
    cout << "sred " << (ppp[1] + ppp[2]) / 2 << " " << ppp[0] << " " << ppp[1] << " " << ppp[2] << " " << ppp[3] << " " << endl;
    //quad = cv::Mat::zeros((ppp[1] + ppp[2]) / 2, (ppp[1] + ppp[2]) / 2, CV_8UC3);
    quad = cv::Mat::zeros(ppp[2], ppp[3], CV_8UC3);
    vector<cv::Point2f> quad_pts;
    quad_pts.push_back(cv::Point2f(0, 0));
    quad_pts.push_back(cv::Point2f(quad.cols, 0));
    quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));
    quad_pts.push_back(cv::Point2f(0, quad.rows));
    Mat transmtx = cv::getPerspectiveTransform(corners, quad_pts);
    warpPerspective(src, quad, transmtx, quad.size());
    imshow("image", dst);
    imshow("povernutoe", quad);
    cout << "homography has done" << endl;
    corners.clear();
    approx.clear();
}
int main() {
    bool checkPicture=false;
    Mat tpl = cv::imread("T4.png");
    Mat cameraFeed;
    Mat for_homography;
    Mat HSV;
    Mat for_triangles;
    VideoCapture capture;
    //capture.open(0);
    //capture.set(CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    //capture.set(CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    capture.open(0);
    while (1) {

        capture.read(cameraFeed);
        capture.read(for_triangles);
        capture.read(for_homography);
        //imshow("Camerafeed", cameraFeed);
        if (for_triangles.empty()||for_homography.empty()) {
            checkPicture = false;
            cout << "gg" << endl;
            capture.open(0);
        }
        else {
            checkPicture = true;
            find_triangles(for_triangles);
        }

        if (check&&checkPicture) {
            cout << "We have good points! Their size:" << good_points.size() << endl;
            if (good_points.size() == 4) {
                cout << "homography active" << endl;
                Homography(for_homography);
            }
            else {

            }
        }
        else {

        }






        waitKey(20);
    }
    destroyAllWindows();
    return 0;
}
