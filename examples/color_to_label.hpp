#ifndef COLOR_TO_LABEL_H
#define COLOR_TO_LABEL_H


#include <opencv2/opencv.hpp>
#include <unordered_map>

struct vec3bcomp{
    bool operator() (const cv::Vec3b& lhs, const cv::Vec3b& rhs) const
        {
            for (int i = 0; i < 3; i++) {
                if(lhs[i]!=rhs[i]){
                    return lhs.val[i]<rhs.val[i];
                }
            }
            return false;
        }
};
typedef std::vector<std::vector<int>> label_matrix;

typedef std::map<cv::Vec3b, int, vec3bcomp> labelindex;

labelindex  get_color_to_label_map_from_dataset(const std::string & dataset_name);
int lookup_label_index(labelindex color_to_label, cv::Vec3b gtVal);
label_matrix labels_from_lblimg(cv::Mat lbl_img, labelindex color_to_label);


#endif /* COLOR_TO_LABEL_H */
