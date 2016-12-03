#include "color_to_label.hpp"


labelindex  get_color_to_label_map_from_dataset(const std::string & dataset_name){
    labelindex color_to_label;

    if (dataset_name == "MSRC") {
        color_to_label[cv::Vec3b(128,0,0)] = 0;
        color_to_label[cv::Vec3b(0,128,0)] = 1;
        color_to_label[cv::Vec3b(128,128,0)] = 2;
        color_to_label[cv::Vec3b(0,0,128)] = 3;
        color_to_label[cv::Vec3b(0,128,128)] = 4;
        color_to_label[cv::Vec3b(128,128,128)] = 5;
        color_to_label[cv::Vec3b(192,0,0)] = 6;
        color_to_label[cv::Vec3b(64,128,0)] = 7;
        color_to_label[cv::Vec3b(192,128,0)] = 8;
        color_to_label[cv::Vec3b(64,0,128)] = 9;
        color_to_label[cv::Vec3b(192,0,128)] = 10;
        color_to_label[cv::Vec3b(64,128,128)] = 11;
        color_to_label[cv::Vec3b(192,128,128)] = 12;
        color_to_label[cv::Vec3b(0,64,0)] = 13;
        color_to_label[cv::Vec3b(128,64,0)] = 14;
        color_to_label[cv::Vec3b(0,192,0)] = 15;
        color_to_label[cv::Vec3b(128,64,128)] = 16;
        color_to_label[cv::Vec3b(0,192,128)] = 17;
        color_to_label[cv::Vec3b(128,192,128)] = 18;
        color_to_label[cv::Vec3b(64,64,0)] = 19;
        color_to_label[cv::Vec3b(192,64,0)] = 20;
        color_to_label[cv::Vec3b(0,0,0)] = 21;

        // Ignored labels
        color_to_label[cv::Vec3b(64,0,0)] = 21;
        color_to_label[cv::Vec3b(128,0,128)] = 21;
    } else if (dataset_name == "Pascal2010") {
        color_to_label[cv::Vec3b(0,0,0)]= 0;
        color_to_label[cv::Vec3b(128,0,0)]= 1;
        color_to_label[cv::Vec3b(0,128,0)]= 2;
        color_to_label[cv::Vec3b(128,128,0)]= 3;
        color_to_label[cv::Vec3b(0,0,128)]= 4;
        color_to_label[cv::Vec3b(128,0,128)]= 5;
        color_to_label[cv::Vec3b(0,128,128)]= 6;
        color_to_label[cv::Vec3b(128,128,128)]= 7;
        color_to_label[cv::Vec3b(64,0,0)]= 8;
        color_to_label[cv::Vec3b(192,0,0)]= 9;
        color_to_label[cv::Vec3b(64,128,0)]= 10;
        color_to_label[cv::Vec3b(192,128,0)]= 11;
        color_to_label[cv::Vec3b(64,0,128)]= 12;
        color_to_label[cv::Vec3b(192,0,128)]= 13;
        color_to_label[cv::Vec3b(64,128,128)]= 14;
        color_to_label[cv::Vec3b(192,128,128)]= 15;
        color_to_label[cv::Vec3b(0,64,0)]= 16;
        color_to_label[cv::Vec3b(128,64,0)]= 17;
        color_to_label[cv::Vec3b(0,192,0)]= 18;
        color_to_label[cv::Vec3b(128,192,0)]= 19;
        color_to_label[cv::Vec3b(0,64,128)]= 20;

        // color_to_label[cv::Vec3b(224, 224, 192)] = 21;
        // color_to_label[cv::Vec3b(255,255,255)] = 21;
        // color_to_label[cv::Vec3b(128,64,128)]= 21;
        // color_to_label[cv::Vec3b(0,192,128)]= 1;
        // color_to_label[cv::Vec3b(128,192,128)]= 1;
        // color_to_label[cv::Vec3b(64,64,0)]= 1;
        // color_to_label[cv::Vec3b(192,64,0)]= 1;
        // color_to_label[cv::Vec3b(64,192,0)]= 1;
        // color_to_label[cv::Vec3b(192,192,0)]= 1;
    }


    return color_to_label;
}

int lookup_label_index(labelindex color_to_label, cv::Vec3b gtVal)
{
    int label=-1;
    try {
        label = color_to_label.at(gtVal);
    } catch( std::out_of_range) {
        //std::cout << gtVal << '\n';
        (void)0;
    }
    if (label != -1) {
        return label;
    } else {
        return 21;
    }
}


label_matrix labels_from_lblimg(cv::Mat lbl_img, labelindex color_to_label){
    label_matrix res(lbl_img.rows, std::vector<int>(lbl_img.cols));

    for(int y = 0; y < lbl_img.rows; ++y)
    {
        for(int x = 0; x < lbl_img.cols; ++x)
        {
            cv::Point p(x,y);
            cv::Vec3b val = lbl_img.at<cv::Vec3b>(p);
            // since OpenCV uses BGR instead of RGB
            std::swap(val[0], val[2]);
            int lbl = lookup_label_index(color_to_label, val);
            res[y][x]= lbl;
        }
    }
    return res;
}
