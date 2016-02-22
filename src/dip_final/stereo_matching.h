#include <iostream>
#include <io.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

std::string GetMatType(const cv::Mat& mat) {
    const int mtype = mat.type();

    switch (mtype) {
        case CV_8UC1:  return "CV_8UC1";
        case CV_8UC2:  return "CV_8UC2";
        case CV_8UC3:  return "CV_8UC3";
        case CV_8UC4:  return "CV_8UC4";

        case CV_8SC1:  return "CV_8SC1";
        case CV_8SC2:  return "CV_8SC2";
        case CV_8SC3:  return "CV_8SC3";
        case CV_8SC4:  return "CV_8SC4";

        case CV_16UC1: return "CV_16UC1";
        case CV_16UC2: return "CV_16UC2";
        case CV_16UC3: return "CV_16UC3";
        case CV_16UC4: return "CV_16UC4";

        case CV_16SC1: return "CV_16SC1";
        case CV_16SC2: return "CV_16SC2";
        case CV_16SC3: return "CV_16SC3";
        case CV_16SC4: return "CV_16SC4";

        case CV_32SC1: return "CV_32SC1";
        case CV_32SC2: return "CV_32SC2";
        case CV_32SC3: return "CV_32SC3";
        case CV_32SC4: return "CV_32SC4";

        case CV_32FC1: return "CV_32FC1";
        case CV_32FC2: return "CV_32FC2";
        case CV_32FC3: return "CV_32FC3";
        case CV_32FC4: return "CV_32FC4";

        case CV_64FC1: return "CV_64FC1";
        case CV_64FC2: return "CV_64FC2";
        case CV_64FC3: return "CV_64FC3";
        case CV_64FC4: return "CV_64FC4";

        default:
            return "Invalid type of matrix!";
    }
}

//  A test program to evaluate the quality of your disparity maps
double evaluate_quality(Mat standard, Mat mine) { 
    int count = 0;
    int error = 0;
    for (int i = 0; i < standard.rows; i++) {
        for (int j = 0; j < standard.cols; j++) {
            error = abs(standard.at<uchar>(i, j) - mine.at<uchar>(i, j));
            if (error > 3) count++;
        }
    }
    return (double) count / (standard.rows*standard.cols);
}

//  A local stereo matching algorithm using “Sum of Squared Diﬀerence(SSD)” as matching cost
void ssd_matching() {
    _finddata_t file, file2;
    long lf, lg;
    if ((lf = _findfirst("ALL-2views\\*", &file)) != -1l) {
        do {
            if (file.attrib == _A_SUBDIR && strcmp(file.name, ".") != 0 && strcmp(file.name, "..") != 0) {
                Mat view1, view5;

                string s = "ALL-2views\\";                                                        //        read view1.png
                s += file.name;
                s += "\\view1.png";
                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    view1 = imread(s, CV_LOAD_IMAGE_GRAYSCALE);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                s = "ALL-2views\\";                                                                 //        read view5.png
                s += file.name;
                s += "\\view5.png";
                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    view5 = imread(s, CV_LOAD_IMAGE_GRAYSCALE);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                int max_offset = 79;
                int window_size = 3;
                int width = view1.cols;
                int height = view1.rows;

                Mat disp1(height, width, CV_8UC1);                                      //      generate disp1.png
                vector< vector<int>> min_ssd;
                for (int i = 0; i < height; ++i) {
                    vector<int> tmp(width, numeric_limits<int>::max());
                    min_ssd.push_back(tmp);
                }

                cout << "generating ssd disp for test :" << file.name << endl;

                for (int offset = 0; offset < max_offset + 1; offset++) {
                    cout << "calculating offset :" << offset << endl;

                    Mat tmp(height, width, CV_8U);

                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < offset; x++) {
                            tmp.at<uchar>(y, x) = view5.at<uchar>(y, x);
                        }
                        for (int x = offset; x < width; x++) {
                            tmp.at<uchar>(y, x) = view5.at<uchar>(y, x - offset);
                        }
                    }

                    //      sd = (view1 - tmp) ** 2
                    Mat sd = Mat::zeros(height, width, CV_32S);
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            sd.at<int>(y, x) = abs(view1.at<uchar>(y, x) - tmp.at<uchar>(y, x));
                            sd.at<int>(y, x) *= sd.at<int>(y, x);
                        }
                    }

                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            int y_start = max(0, y - window_size);
                            int y_end = min(height, y + window_size + 1);
                            int x_start = max(0, x - window_size);
                            int x_end = min(width, x + window_size + 1);
                            int ssd = 0;

                            //      the matching cost function
                            for (int y2 = 0; y2 < y_end - y_start; y2++) {
                                for (int x2 = 0; x2 < x_end - x_start; x2++) {
                                    ssd += sd.at<int>(y2 + y_start, x2 + x_start);
                                }
                            }

                            if (ssd < min_ssd[y][x]) {
                                min_ssd[y][x] = ssd;
                                disp1.at<uchar>(y, x) = (uchar) (offset * 3);
                            }
                        }
                    }
                }

                imshow("disp", disp1);
                s = "ALL-2views\\";                                                             //            save disp1.png
                s += file.name;
                s += "\\";
                s += file.name;
                s += "_disp1_SSD.png";
                imwrite(s, disp1);
                waitKey(100);

                Mat disp5(height, width, CV_8UC1);                                      //      generate disp5.png
                min_ssd.clear();
                for (int i = 0; i < height; ++i) {
                    vector<int> tmp(width, numeric_limits<int>::max());
                    min_ssd.push_back(tmp);
                }

                cout << "generating ssd disp for test :" << file.name << endl;

                for (int offset = 0; offset < max_offset + 1; offset++) {
                    cout << "calculating offset :" << offset << endl;

                    Mat tmp(height, width, CV_8U);
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width - offset; x++) {
                            tmp.at<uchar>(y, x) = view1.at<uchar>(y, x + offset);
                        }
                        for (int x = width - offset; x < width; x++) {
                            tmp.at<uchar>(y, x) = view1.at<uchar>(y, x);
                        }
                    }

                    //      sd = (view5 - tmp) ** 2
                    Mat sd = Mat::zeros(height, width, CV_32S);
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            sd.at<int>(y, x) = abs(view5.at<uchar>(y, x) - tmp.at<uchar>(y, x));
                            sd.at<int>(y, x) *= sd.at<int>(y, x);
                        }
                    }

                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            int y_start = max(0, y - window_size);
                            int y_end = min(height, y + window_size + 1);
                            int x_start = max(0, x - window_size);
                            int x_end = min(width, x + window_size + 1);
                            int ssd = 0;

                            //  the matching cost function
                            for (int y2 = 0; y2 < y_end - y_start; y2++) {
                                for (int x2 = 0; x2 < x_end - x_start; x2++) {
                                    ssd += sd.at<int>(y2 + y_start, x2 + x_start);
                                }
                            }

                            if (ssd < min_ssd[y][x]) {
                                min_ssd[y][x] = ssd;
                                disp5.at<uchar>(y, x) = (uchar) (offset * 3);
                            }
                        }
                    }
                }

                imshow("disp", disp5);
                s = "ALL-2views\\";                                                             //            save disp5.png
                s += file.name;
                s += "\\";
                s += file.name;
                s += "_disp5_SSD.png";
                imwrite(s, disp5);
                waitKey(100);
            }
        } while (_findnext(lf, &file) == 0);
        _findclose(lf);
    }
}

void evaluate_ssd() {
    _finddata_t file, file2;
    long lf, lg;
    if ((lf = _findfirst("ALL-2views\\*", &file)) != -1l) {
        do {
            if (file.attrib == _A_SUBDIR && strcmp(file.name, ".") != 0 && strcmp(file.name, "..") != 0) {
                Mat disp1, disp5;
                Mat mine1, mine5;

                string s = "ALL-2views\\";                                                        //        read disp1.png
                s += file.name;
                s += "\\disp1.png";
                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    disp1 = imread(s);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                s = "ALL-2views\\";                                                                 //        read disp5.png
                s += file.name;
                s += "\\disp5.png";
                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    disp5 = imread(s);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                s = "ALL-2views\\";                                                                 //        read mine1
                s += file.name;
                s += "\\";
                s += file.name;
                s += "_disp1_SSD.png";

                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    mine1 = imread(s);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                s = "ALL-2views\\";                                                                 //        read mine5
                s += file.name;
                s += "\\";
                s += file.name;
                s += "_disp5_SSD.png";

                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    mine5 = imread(s);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                cout << file.name << ":   " << endl;

                cout << "The percentage of bad pixels in my ssd disp1 : " << evaluate_quality(disp1, mine1) << endl;
                cout << "The percentage of bad pixels in my ssd disp5 : " << evaluate_quality(disp5, mine5) << endl;
            }
        } while (_findnext(lf, &file) == 0);
        _findclose(lf);
    }
}

//  A local stereo matching algorithm using “Normalized Cross Correlation(NCC)” as matching cost.
void ncc_matching() {
    _finddata_t file, file2;
    long lf, lg;
    if ((lf = _findfirst("ALL-2views\\*", &file)) != -1l) {
        do {
            if (file.attrib == _A_SUBDIR && strcmp(file.name, ".") != 0 && strcmp(file.name, "..") != 0) {
                Mat view1, view5;

                string s = "ALL-2views\\";                                                        //        read view1.png
                s += file.name;
                s += "\\view1.png";
                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    view1 = imread(s, CV_LOAD_IMAGE_GRAYSCALE);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                s = "ALL-2views\\";                                                                 //        read view5.png
                s += file.name;
                s += "\\view5.png";
                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    view5 = imread(s, CV_LOAD_IMAGE_GRAYSCALE);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                int max_offset = 79;
                int window_size = 3;
                int width = view1.cols;
                int height = view1.rows;

                Mat disp1(height, width, CV_8UC1);                                      //      generate disp1.png
                vector< vector<double>> max_ncc;
                for (int i = 0; i < height; ++i) {
                    vector<double> tmp(width, -numeric_limits<double>::max());
                    max_ncc.push_back(tmp);
                }

                cout << "generating ncc disp for test :" << file.name << endl;

                for (int offset = 0; offset < max_offset + 1; offset++) {
                    cout << "calculating offset :" << offset << endl;

                    Mat tmp(height, width, CV_8U);

                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < offset; x++) {
                            tmp.at<uchar>(y, x) = view5.at<uchar>(y, x);
                        }
                        for (int x = offset; x < width; x++) {
                            tmp.at<uchar>(y, x) = view5.at<uchar>(y, x - offset);
                        }
                    }

                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            int y_start = max(0, y - window_size);
                            int y_end = min(height, y + window_size + 1);
                            int x_start = max(0, x - window_size);
                            int x_end = min(width, x + window_size + 1);
                            double SRL = 0, SRR = 0, SLL = 0, SR = 0, SL = 0;
                            double SIZE = (y_end - y_start) * (x_end - x_start);

                            for (int y2 = y_start; y2 < y_end; y2++) {
                                for (int x2 = x_start; x2 < x_end; x2++) {
                                    SRL += view1.at<uchar>(y2, x2) * tmp.at<uchar>(y2, x2);
                                    SRR += tmp.at<uchar>(y2, x2) * tmp.at<uchar>(y2, x2);
                                    SLL += view1.at<uchar>(y2, x2) * view1.at<uchar>(y2, x2);
                                    SR += tmp.at<uchar>(y2, x2);
                                    SL += view1.at<uchar>(y2, x2);
                                }
                            }

                            //       formula of the NCC matching cost
                            double sum_ncc = (SRL - SR*SL / SIZE) / sqrt((SRR - SR*SR / SIZE)*(SLL - SL*SL / SIZE));

                            if (sum_ncc > max_ncc[y][x]) {
                                max_ncc[y][x] = sum_ncc;
                                disp1.at<uchar>(y, x) = (uchar) (offset * 3);
                            }
                        }
                    }
                }

                imshow("disp", disp1);
                s = "ALL-2views\\";                                                             //            save disp1.png
                s += file.name;
                s += "\\";
                s += file.name;
                s += "_disp1_NCC.png";
                imwrite(s, disp1);
                waitKey(100);
            }
        } while (_findnext(lf, &file) == 0);
        _findclose(lf);
    }
}

void evaluate_ncc() {
    _finddata_t file, file2;
    long lf, lg;
    if ((lf = _findfirst("ALL-2views\\*", &file)) != -1l) {
        do {
            if (file.attrib == _A_SUBDIR && strcmp(file.name, ".") != 0 && strcmp(file.name, "..") != 0) {
                Mat disp1, mine1;

                string s = "ALL-2views\\";                                                        //        read disp1.png
                s += file.name;
                s += "\\disp1.png";
                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    disp1 = imread(s);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                s = "ALL-2views\\";                                                                 //        read mine1
                s += file.name;
                s += "\\";
                s += file.name;
                s += "_disp1_NCC.png";

                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    mine1 = imread(s);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                cout << file.name << ":   " << endl;

                cout << "The percentage of bad pixels in my ncc disp1 : " << evaluate_quality(disp1, mine1) << endl;
            }
        } while (_findnext(lf, &file) == 0);
        _findclose(lf);
    }
}

//  Add a small constant amount of intensity (e.g. 10) to all right eye images
void right_eye() {
    _finddata_t file, file2;
    long lf, lg;
    if ((lf = _findfirst("ALL-2views\\*", &file)) != -1l) {
        do {
            if (file.attrib == _A_SUBDIR && strcmp(file.name, ".") != 0 && strcmp(file.name, "..") != 0) {
                Mat view1, view5;

                string s = "ALL-2views\\";                                                        //        read view1.png
                s += file.name;
                s += "\\view1.png";
                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    view1 = imread(s, CV_LOAD_IMAGE_GRAYSCALE);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                s = "ALL-2views\\";                                                                 //        read view5.png
                s += file.name;
                s += "\\view5.png";
                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    view5 = imread(s, CV_LOAD_IMAGE_GRAYSCALE);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                int max_offset = 79;
                int window_size = 3;
                int width = view1.cols;
                int height = view1.rows;

                //          right eyed strengthen
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        view5.at<uchar>(y, x) += 10;
                    }
                }

                Mat disp1(height, width, CV_8UC1);                                      //      generate disp1.png
                Mat disp5(height, width, CV_8UC1);                                      //      generate disp5.png

                vector< vector<int>> min_ssd1;
                vector< vector<int>> min_ssd5;

                for (int i = 0; i < height; ++i) {
                    vector<int> tmp(width, numeric_limits<int>::max());
                    min_ssd1.push_back(tmp);
                    min_ssd5.push_back(tmp);
                }

                cout << "generating right eyed ssd disp for test :" << file.name << endl;

                for (int offset = 0; offset < max_offset + 1; offset++) {
                    cout << "calculating offset :" << offset << endl;

                    Mat tmp1(height, width, CV_8U);
                    Mat tmp5(height, width, CV_8U);

                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < offset; x++) {
                            tmp1.at<uchar>(y, x) = view5.at<uchar>(y, x);
                        }
                        for (int x = offset; x < width; x++) {
                            tmp1.at<uchar>(y, x) = view5.at<uchar>(y, x - offset);
                        }
                    }
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width - offset; x++) {
                            tmp5.at<uchar>(y, x) = view1.at<uchar>(y, x + offset);
                        }
                        for (int x = width - offset; x < width; x++) {
                            tmp5.at<uchar>(y, x) = view1.at<uchar>(y, x);
                        }
                    }

                    Mat sd1 = Mat::zeros(height, width, CV_32S);
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            sd1.at<int>(y, x) = abs(view1.at<uchar>(y, x) - tmp1.at<uchar>(y, x));
                            sd1.at<int>(y, x) *= sd1.at<int>(y, x);
                        }
                    }
                    Mat sd5 = Mat::zeros(height, width, CV_32S);
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            sd5.at<int>(y, x) = abs(view5.at<uchar>(y, x) - tmp5.at<uchar>(y, x));
                            sd5.at<int>(y, x) *= sd5.at<int>(y, x);
                        }
                    }

                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            int y_start = max(0, y - window_size);
                            int y_end = min(height, y + window_size + 1);
                            int x_start = max(0, x - window_size);
                            int x_end = min(width, x + window_size + 1);
                            int ssd1 = 0;
                            int ssd5 = 0;

                            for (int y2 = 0; y2 < y_end - y_start; y2++) {
                                for (int x2 = 0; x2 < x_end - x_start; x2++) {
                                    ssd1 += sd1.at<int>(y2 + y_start, x2 + x_start);
                                    ssd5 += sd5.at<int>(y2 + y_start, x2 + x_start);
                                }
                            }

                            if (ssd1 < min_ssd1[y][x]) {
                                min_ssd1[y][x] = ssd1;
                                disp1.at<uchar>(y, x) = (uchar) (offset * 3);
                            }
                            if (ssd5 < min_ssd5[y][x]) {
                                min_ssd5[y][x] = ssd5;
                                disp5.at<uchar>(y, x) = (uchar) (offset * 3);
                            }
                        }
                    }
                }

                imshow("disp", disp1);
                s = "ALL-2views\\";                                                             //            save disp1.png
                s += file.name;
                s += "\\";
                s += file.name;
                s += "_disp1_right_eye_SSD.png";
                imwrite(s, disp1);
                waitKey(100);

                imshow("disp", disp5);
                s = "ALL-2views\\";                                                             //            save disp5.png
                s += file.name;
                s += "\\";
                s += file.name;
                s += "_disp5_right_eye_SSD.png";
                imwrite(s, disp5);
                waitKey(100);

                Mat disp3(height, width, CV_8UC1);                                      //      generate disp3.png
                vector< vector<double>> max_ncc;
                for (int i = 0; i < height; ++i) {
                    vector<double> tmp(width, -numeric_limits<double>::max());
                    max_ncc.push_back(tmp);
                }

                cout << "generating right eyed ncc disp for test :" << file.name << endl;

                for (int offset = 0; offset < max_offset + 1; offset++) {
                    cout << "calculating offset :" << offset << endl;

                    Mat tmp(height, width, CV_8U);

                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < offset; x++) {
                            tmp.at<uchar>(y, x) = view5.at<uchar>(y, x);
                        }
                        for (int x = offset; x < width; x++) {
                            tmp.at<uchar>(y, x) = view5.at<uchar>(y, x - offset);
                        }
                    }

                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            int y_start = max(0, y - window_size);
                            int y_end = min(height, y + window_size + 1);
                            int x_start = max(0, x - window_size);
                            int x_end = min(width, x + window_size + 1);
                            double SRL = 0, SRR = 0, SLL = 0, SR = 0, SL = 0;
                            double SIZE = (y_end - y_start) * (x_end - x_start);

                            for (int y2 = y_start; y2 < y_end; y2++) {
                                for (int x2 = x_start; x2 < x_end; x2++) {
                                    SRL += view1.at<uchar>(y2, x2) * tmp.at<uchar>(y2, x2);
                                    SRR += tmp.at<uchar>(y2, x2) * tmp.at<uchar>(y2, x2);
                                    SLL += view1.at<uchar>(y2, x2) * view1.at<uchar>(y2, x2);
                                    SR += tmp.at<uchar>(y2, x2);
                                    SL += view1.at<uchar>(y2, x2);
                                }
                            }

                            double sum_ncc = (SRL - SR*SL / SIZE) / sqrt((SRR - SR*SR / SIZE)*(SLL - SL*SL / SIZE));

                            if (sum_ncc > max_ncc[y][x]) {
                                max_ncc[y][x] = sum_ncc;
                                disp3.at<uchar>(y, x) = (uchar) (offset * 3);
                            }
                        }
                    }
                }

                imshow("disp", disp3);
                s = "ALL-2views\\";                                                             //            save disp3.png
                s += file.name;
                s += "\\";
                s += file.name;
                s += "_disp1_right_eye_NCC.png";
                imwrite(s, disp3);
                waitKey(100);
            }
        } while (_findnext(lf, &file) == 0);
        _findclose(lf);
    }
}

void evaluate_right_eye() {
    _finddata_t file, file2;
    long lf, lg;
    if ((lf = _findfirst("ALL-2views\\*", &file)) != -1l) {
        do {
            if (file.attrib == _A_SUBDIR && strcmp(file.name, ".") != 0 && strcmp(file.name, "..") != 0) {
                Mat disp1, disp5;
                Mat mine1, mine5, mine3;

                string s = "ALL-2views\\";                                                        //        read disp1.png
                s += file.name;
                s += "\\disp1.png";
                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    disp1 = imread(s);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                s = "ALL-2views\\";                                                                 //        read disp5.png
                s += file.name;
                s += "\\disp5.png";
                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    disp5 = imread(s);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                s = "ALL-2views\\";                                                                 //        read mine1
                s += file.name;
                s += "\\";
                s += file.name;
                s += "_disp1_right_eye_SSD.png";

                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    mine1 = imread(s);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                s = "ALL-2views\\";                                                                 //        read mine5
                s += file.name;
                s += "\\";
                s += file.name;
                s += "_disp5_right_eye_SSD.png";

                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    mine5 = imread(s);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                s = "ALL-2views\\";                                                                 //        read mine3.png
                s += file.name;
                s += "\\";
                s += file.name;
                s += "_disp1_right_eye_NCC.png";

                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    mine3 = imread(s);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                cout << file.name << ":   " << endl;

                cout << "The percentage of bad pixels in my right eyed ssd disp1 : " << evaluate_quality(disp1, mine1) << endl;
                cout << "The percentage of bad pixels in my right eyed ssd disp5 : " << evaluate_quality(disp5, mine5) << endl;
                cout << "The percentage of bad pixels in my right eyed ncc disp1 : " << evaluate_quality(disp1, mine3) << endl;
            }
        } while (_findnext(lf, &file) == 0);
        _findclose(lf);
    }
}

//  Cost aggregation using Adaptive Support Weight (ASW) 
void asw() {
    _finddata_t file, file2;
    long lf, lg;
    if ((lf = _findfirst("ALL-2views\\*", &file)) != -1l) {
        do {
            if (file.attrib == _A_SUBDIR && strcmp(file.name, ".") != 0 && strcmp(file.name, "..") != 0) {
                Mat view1, view5;
                Mat lab1, lab5;

                string s = "ALL-2views\\";                                                        //        read view1.png
                s += file.name;
                s += "\\view1.png";
                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    view1 = imread(s);
                    cvtColor(view1, lab1, CV_RGB2Lab);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                s = "ALL-2views\\";                                                                 //        read view5.png
                s += file.name;
                s += "\\view5.png";
                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    view5 = imread(s);
                    cvtColor(view5, lab5, CV_RGB2Lab);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                int max_offset = 79;
                int window_size = 8;
                double k = 10, gamma_c = 7, gamma_g = 20; // ASW parameters
                int width = view1.cols;
                int height = view1.rows;

                Mat disp1(height, width, CV_8UC1);                                      //      generate disp1.png
                vector< vector<double>> min_asw;
                for (int i = 0; i < height; ++i) {
                    vector<double> tmp(width, numeric_limits<double>::max());
                    min_asw.push_back(tmp);
                }

                cout << "generating asw disp for test :" << file.name << endl;

                for (int offset = 0; offset < max_offset + 1; offset++) {
                    cout << "calculating offset :" << offset << endl;

                    Mat tmp(height, width, CV_8UC3);
                    Mat rgb5(height, width, CV_8UC3);

                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < offset; x++) {
                            tmp.at<Vec3b>(y, x) = lab5.at<Vec3b>(y, x);
                            rgb5.at<Vec3b>(y, x) = view5.at<Vec3b>(y, x);
                        }
                        for (int x = offset; x < width; x++) {
                            tmp.at<Vec3b>(y, x) = lab5.at<Vec3b>(y, x - offset);
                            rgb5.at<Vec3b>(y, x) = view5.at<Vec3b>(y, x - offset);
                        }
                    }

                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            int y_start = max(0, y - window_size);
                            int y_end = min(height, y + window_size + 1);
                            int x_start = max(0, x - window_size);
                            int x_end = min(width, x + window_size + 1);
                            int ssd = 0;
                            double sum1 = 0;
                            double sum2 = 0;

                            for (int y2 = y_start; y2 < y_end; y2++) {
                                for (int x2 = x_start; x2 < x_end; x2++) {
                                    //      ∆cpq = (Lp −Lq)2 + (ap −aq)2 + (bp −bq)2
                                    double delta_c1 = 0;
                                    delta_c1 += (lab1.at<Vec3b>(y2, x2)[0] - lab1.at<Vec3b>(y, x)[0]) * (lab1.at<Vec3b>(y2, x2)[0] - lab1.at<Vec3b>(y, x)[0]);
                                    delta_c1 += (lab1.at<Vec3b>(y2, x2)[1] - lab1.at<Vec3b>(y, x)[1]) * (lab1.at<Vec3b>(y2, x2)[1] - lab1.at<Vec3b>(y, x)[1]);
                                    delta_c1 += (lab1.at<Vec3b>(y2, x2)[2] - lab1.at<Vec3b>(y, x)[2]) * (lab1.at<Vec3b>(y2, x2)[2] - lab1.at<Vec3b>(y, x)[2]);
                                    delta_c1 = sqrt(delta_c1);

                                    double delta_c2 = 0;
                                    delta_c2 += (tmp.at<Vec3b>(y2, x2)[0] - tmp.at<Vec3b>(y, x)[0]) * (tmp.at<Vec3b>(y2, x2)[0] - tmp.at<Vec3b>(y, x)[0]);
                                    delta_c2 += (tmp.at<Vec3b>(y2, x2)[1] - tmp.at<Vec3b>(y, x)[1]) * (tmp.at<Vec3b>(y2, x2)[1] - tmp.at<Vec3b>(y, x)[1]);
                                    delta_c2 += (tmp.at<Vec3b>(y2, x2)[2] - tmp.at<Vec3b>(y, x)[2]) * (tmp.at<Vec3b>(y2, x2)[2] - tmp.at<Vec3b>(y, x)[2]);
                                    delta_c2 = sqrt(delta_c2);

                                    //      ∆gpq
                                    double delta_g = sqrt((y2 - y) * (y2 - y) + (x2 - x) * (x2 - x));

                                    //      w(p,q)=k·exp[−(∆cpq/γc +∆gpq/γp)]
                                    double w1 = k * exp(-(delta_c1 / gamma_c + delta_g / gamma_g));
                                    double w2 = k * exp(-(delta_c2 / gamma_c + delta_g / gamma_g));

                                    //      e0(q,¯ qd)=  c∈{r,g,b}| Ic(q)−Ic(¯ qd) |
                                    double e = 0;
                                    e += (view1.at<Vec3b>(y2, x2)[0] - rgb5.at<Vec3b>(y2, x2)[0]) * (view1.at<Vec3b>(y2, x2)[0] - rgb5.at<Vec3b>(y2, x2)[0]);
                                    e += (view1.at<Vec3b>(y2, x2)[1] - rgb5.at<Vec3b>(y2, x2)[1]) * (view1.at<Vec3b>(y2, x2)[1] - rgb5.at<Vec3b>(y2, x2)[1]);
                                    e += (view1.at<Vec3b>(y2, x2)[2] - rgb5.at<Vec3b>(y2, x2)[2]) * (view1.at<Vec3b>(y2, x2)[2] - rgb5.at<Vec3b>(y2, x2)[2]);
                                    e = sqrt(e);

                                    sum1 += w1*w2*e;
                                    sum2 += w1*w2;
                                }
                            }

                            double sum_asw = sum1 / sum2;

                            if (sum_asw < min_asw[y][x]) {
                                min_asw[y][x] = sum_asw;
                                disp1.at<uchar>(y, x) = (uchar) (offset * 3);
                            }
                        }
                    }
                }

                imshow("disp", disp1);
                s = "ALL-2views\\";                                                             //            save disp1.png
                s += file.name;
                s += "\\";
                s += file.name;
                s += "_disp1_ASW.png";
                imwrite(s, disp1);
                waitKey(100);
            }
        } while (_findnext(lf, &file) == 0);
        _findclose(lf);
    }
}

void evaluate_asw() {
    _finddata_t file, file2;
    long lf, lg;
    if ((lf = _findfirst("ALL-2views\\*", &file)) != -1l) {
        do {
            if (file.attrib == _A_SUBDIR && strcmp(file.name, ".") != 0 && strcmp(file.name, "..") != 0) {
                Mat disp1, mine1;

                string s = "ALL-2views\\";                                                        //        read disp1.png
                s += file.name;
                s += "\\disp1.png";
                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    disp1 = imread(s);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                s = "ALL-2views\\";                                                                 //        read mine1.png
                s += file.name;
                s += "\\";
                s += file.name;
                s += "_disp1_ASW.png";

                if ((lg = _findfirst(s.c_str(), &file2)) != -1l) {
                    mine1 = imread(s);
                    //imshow(file2.name, img);
                    _findclose(lg);
                }

                cout << file.name << ":   " << endl;

                cout << "The percentage of bad pixels in my asw disp1 : " << evaluate_quality(disp1, mine1) << endl;
            }
        } while (_findnext(lf, &file) == 0);
        _findclose(lf);
    }
}