#include "file_storage.hpp"
#include "probimage.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sys/stat.h>


// Directory and file stuff
void make_dir(std::string dir_name){
    struct stat resdir_stat;
    if (stat(dir_name.c_str(), &resdir_stat) == -1) {
        mkdir(dir_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
}


bool file_exist(std::string file_path){
    struct stat path_stat;
    return stat(file_path.c_str(),&path_stat)==0;
}


static inline std::string &rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}


void split_string(const std::string &s, const char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}

std::string stringreplace(std::string s,
                          std::string toReplace,
                          std::string replaceWith)
{
    if (s.find(toReplace) != std::string::npos){
        return(s.replace(s.find(toReplace), toReplace.length(), replaceWith));
    }

    return s;
}


Dataset::Dataset(std::string path_to_images,
                 std::string path_to_unaries,
                 std::string path_to_ground_truths,
                 std::string path_to_root,
                 std::string image_format,
                 std::string ground_truth_format,
                 std::string name): path_to_images(path_to_images),
                                    path_to_unaries(path_to_unaries),
                                    path_to_ground_truths(path_to_ground_truths),
                                    path_to_root(path_to_root),
                                    image_format(image_format),
                                    ground_truth_format(ground_truth_format),
                                    name(name){}

std::string Dataset::get_unaries_path(const std::string & image_name){
    std::string unaries_path = path_to_unaries + image_name;
    unaries_path = unaries_path + ".c_unary";
    return unaries_path;
}

std::string Dataset::get_image_path(const std::string & image_name){
    std::string image_path = path_to_images + image_name;
    image_path = image_path + image_format;
    return image_path;
}

std::string Dataset::get_ground_truth_path(const std::string & image_name){
    std::string gt_path = path_to_ground_truths + image_name;
    gt_path = gt_path + ground_truth_format;
    return gt_path;
}

std::vector<std::string> Dataset::get_all_split_files(const std::string & split)
{
    std::string path_to_split = path_to_root + "split/" + split+ ".txt";

    std::vector<std::string> split_images;
    std::string next_img_name;
    std::ifstream file(path_to_split.c_str());


    while(getline(file, next_img_name)){
        next_img_name = rtrim(next_img_name);
        next_img_name = stringreplace(next_img_name, ".bmp", ""); // Cleanup the name in MSRC
        split_images.push_back(next_img_name);
    }
    return split_images;
}

Dataset get_dataset_by_name(const std::string & dataset_name){
    if (dataset_name == "MSRC") {
        return Dataset("/data/MSRC/MSRC_ObjCategImageDatabase_v2/Images/",
                       "/data/MSRC/texton_unaries/",
                       "/data/MSRC/MSRC_ObjCategImageDatabase_v2/GroundTruth",
                       "/data/MSRC/",
                       ".bmp",
                       "_GT.bmp",
                       "MSRC");
    }
    else if (dataset_name == "Pascal2010") {
        return Dataset("/data/PascalVOC2010/JPEGImages/",
                       "/data/PascalVOC2010/logit_unaries/",
                       "/data/PascalVOC2010/SegmentationClass/",
                       "/data/PascalVOC2010/",
                       ".jpg",
                       ".png",
                       "Pascal2010");
    }
    // Add some possible other datasets
    else if (dataset_name == "Pascal2010_2") {
        return Dataset("/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/PascalVOC2010/JPEGImages/",
                       "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/PascalVOC2010/logit_unaries/",
                       "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/PascalVOC2010/SegmentationClass/",
                       "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/PascalVOC2010/",
                       ".jpg",
                       ".png",
                       "Pascal2010");
    }
    else if (dataset_name == "MSRC_2") {
        return Dataset("/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/MSRC/MSRC_ObjCategImageDatabase_v2/Images/",
                       "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/MSRC/texton_unaries/",
                       "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/MSRC/MSRC_ObjCategImageDatabase_v2/GroundTruth",
                       "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/MSRC/",
                       ".bmp",
                       "_GT.bmp",
                       "MSRC");
    }
    // Add some possible other datasets
    else if (dataset_name == "Pascal2010_3") {
        return Dataset("/media/ajanthan/sheep/Ajanthan/data/PascalVOC2010/JPEGImages/",
                       "/media/ajanthan/sheep/Ajanthan/data/PascalVOC2010/logit_unaries/",
                       "/media/ajanthan/sheep/Ajanthan/data/PascalVOC2010/SegmentationClass/",
                       "/media/ajanthan/sheep/Ajanthan/data/PascalVOC2010/",
                       ".jpg",
                       ".png",
                       "Pascal2010");
    }
    else if (dataset_name == "MSRC_3") {
        return Dataset("/media/ajanthan/sheep/Ajanthan/data/MSRC/MSRC_ObjCategImageDatabase_v2/Images/",
                       "/media/ajanthan/sheep/Ajanthan/data/MSRC/texton_unaries/",
                       "/media/ajanthan/sheep/Ajanthan/data/MSRC/MSRC_ObjCategImageDatabase_v2/GroundTruth",
                       "/media/ajanthan/sheep/Ajanthan/data/MSRC/",
                       ".bmp",
                       "_GT.bmp",
                       "MSRC");
    }
    else if (dataset_name == "Pascal2010_4") {
        return Dataset("/Ajanthan/data/PascalVOC2010/JPEGImages/",
                       "/Ajanthan/data/PascalVOC2010/logit_unaries/",
                       "/Ajanthan/data/PascalVOC2010/SegmentationClass/",
                       "/Ajanthan/data/PascalVOC2010/",
                       ".jpg",
                       ".png",
                       "Pascal2010");
    }
    else if (dataset_name == "MSRC_4") {
        return Dataset("/Ajanthan/data/MSRC/MSRC_ObjCategImageDatabase_v2/Images/",
                       "/Ajanthan/data/MSRC/texton_unaries/",
                       "/Ajanthan/data/MSRC/MSRC_ObjCategImageDatabase_v2/GroundTruth",
                       "/Ajanthan/data/MSRC/",
                       ".bmp",
                       "_GT.bmp",
                       "MSRC");
    }
}

std::string get_output_path(const std::string & path_to_results_folder, const std::string & image_name){
    std::string output_path = path_to_results_folder + image_name + ".png";
    return output_path;
}

unsigned char * load_image( const std::string & path_to_image, img_size & size){
    cv::Mat img = cv::imread(path_to_image);

    if(size.height != img.rows || size.width != img.cols) {
        std::cout << "Dimension doesn't correspond to unaries" << std::endl;
        if (size.height == -1) {
            size.height = img.rows;
            std::cout << "Adjusting height because was undefined" << '\n';
        }
        if (size.width == -1) {
            size.width = img.cols;
            std::cout << "Adjusting width because was undefined" << '\n';
        }
    }

    unsigned char * char_img = new unsigned char[size.width*size.height*3];
    for (int j=0; j < size.height; j++) {
        for (int i=0; i < size.width; i++) {
            cv::Vec3b intensity = img.at<cv::Vec3b>(j,i); // this comes in BGR
            char_img[(i+j*size.width)*3+0] = intensity.val[2];
            char_img[(i+j*size.width)*3+1] = intensity.val[1];
            char_img[(i+j*size.width)*3+2] = intensity.val[0];
        }
    }

    return char_img;
}

unsigned char * load_rescaled_image( const std::string & path_to_image, img_size & size, int imskip){
    assert(imskip > 1);
    if (imskip < 1) imskip = 1;
    cv::Mat img = cv::imread(path_to_image);

    img_size size2 = {-1, -1};
    if(size2.height != img.rows || size2.width != img.cols) {
        std::cout << "Dimension doesn't correspond to unaries" << std::endl;
        if (size2.height == -1) {
            size2.height = img.rows;
            std::cout << "Adjusting height because was undefined" << '\n';
        }
        if (size2.width == -1) {
            size2.width = img.cols;
            std::cout << "Adjusting width because was undefined" << '\n';
        }
    }
    size.width = size2.width / imskip;
    size.height = size2.height / imskip;

    unsigned char * char_img = new unsigned char[size.width*size.height*3]();
    for (int j=0; j < size.height; j++) {
        if (j % imskip != 0) continue;
        for (int i=0; i < size.width; i++) {
            if (i % imskip != 0) continue;
            cv::Vec3b intensity = img.at<cv::Vec3b>(j,i); // this comes in BGR
            int ii = i/imskip; 
            int jj = j/imskip;
            char_img[(ii+jj*size.width)*3+0] = intensity.val[2];
            char_img[(ii+jj*size.width)*3+1] = intensity.val[1];
            char_img[(ii+jj*size.width)*3+2] = intensity.val[0];
        }
    }

    return char_img;
}

MatrixXf load_unary( const std::string & path_to_unary, img_size& size, int max_label) {

    ProbImage texton;
    texton.decompress(path_to_unary.c_str());
    texton.boostToProb();

    if(size.width<=0 && size.height<=0) {
        size = {texton.width(), texton.height()};
    }
    if(max_label<=0) {
        max_label = texton.depth();
    }

    MatrixXf unaries( max_label, size.width * size.height);
    int i,j,k;
    for(i=0; i<size.height; ++i){
        for(j=0; j<size.width; ++j){
            for(k=0; k<max_label; ++k){
                // careful with the index position, the operator takes
                // x (along width), then y (along height)

                // Also note that these are probabilities, what we
                // want are unaries, so we need to
                unaries(k, i*size.width + j) = -log( texton(j,i,k));
            }
        }
    }

    return unaries;
}

Matrix<short,Dynamic,1> load_labeling(const std::string & path_to_labels, const std::string & dataset_name,
                                      img_size& size){
    Matrix<short,Dynamic,1> labeling(size.width * size.height);

    cv::Mat img = cv::imread(path_to_labels);
    if(size.height != img.rows || size.width != img.cols) {
        std::cout << "Dimension doesn't correspond to labeling" << std::endl;
    }

    labelindex lbl_idx = get_color_to_label_map_from_dataset(dataset_name);

    for (int j=0; j < size.height; j++) {
        for (int i=0; i < size.width; i++) {
            cv::Vec3b intensity = img.at<cv::Vec3b>(j,i); // this comes in BGR
            labeling(j*size.width+i) = lookup_label_index(lbl_idx, intensity);
        }
    }

    return labeling;
}

label_matrix get_label_matrix(const MatrixXf & estimates, const img_size & size){
    label_matrix res(size.height, std::vector<int>(size.width));
    for(int i=0; i<estimates.cols(); ++i) {
        int lbl;
        estimates.col(i).maxCoeff( &lbl);
        int col = i % size.width;
        int row = (i - col)/size.width;
        res[row][col] = lbl;
    }
    return res;
}

label_matrix load_label_matrix(const std::string & path_to_labels, const std::string & dataset_name){
    cv::Mat img = cv::imread(path_to_labels);
    labelindex color_to_label = get_color_to_label_map_from_dataset(dataset_name);
    return labels_from_lblimg(img, color_to_label);
}

void save_map(const MatrixXf & estimates, const img_size & size, const std::string & path_to_output, const std::string & dataset_name) {
    std::vector<short> labeling(estimates.cols());

    // MAP estimation
    for(int i=0; i<estimates.cols(); ++i) {
        int lbl;
        estimates.col(i).maxCoeff( &lbl);
        labeling[i] = lbl;
    }
    cv::Mat img(size.height, size.width, CV_8UC3);
    cv::Vec3b intensity;
    if(dataset_name == "Stereo_special") {
        // Make the image
        int max_label = *std::max_element(labeling.begin(), labeling.end());
        for(int i=0; i<estimates.cols(); ++i) {
            intensity[2] = 255.0*labeling[i]/max_label;
            intensity[1] = 255.0*labeling[i]/max_label;
            intensity[0] = 255.0*labeling[i]/max_label;

            int col = i % size.width;
            int row = (i - col)/size.width;
            img.at<cv::Vec3b>(row, col) = intensity;
        }
    } else {
        const unsigned char*  legend;
        if (dataset_name == "MSRC") {
            legend = MSRC_legend;
        } else if (dataset_name == "Pascal2010") {
            legend = Pascal_legend;
        } else {
            legend = Stereo_legend;
        }

        // Make the image
        for(int i=0; i<estimates.cols(); ++i) {
            intensity[2] = legend[3*labeling[i]];
            intensity[1] = legend[3*labeling[i] + 1];
            intensity[0] = legend[3*labeling[i] + 2];

            int col = i % size.width;
            int row = (i - col)/size.width;
            img.at<cv::Vec3b>(row, col) = intensity;
        }
    }

    cv::imwrite(path_to_output, img);
}

void save_less_confident_pixels(const MatrixXf & estimates, const std::vector<int> & pI, const img_size & size, 
        const std::string & path_to_output, const std::string & dataset_name) {
    std::vector<short> labeling(estimates.cols());

    std::string out_file_name = path_to_output;
    out_file_name.replace(out_file_name.end()-4, out_file_name.end(),"_lcf.png");
    // MAP estimation
    for(int i=0; i<estimates.cols(); ++i) {
        int lbl;
        estimates.col(i).maxCoeff( &lbl);
        labeling[i] = lbl;
    }
    cv::Mat img(size.height, size.width, CV_8UC3);
    cv::Vec3b intensity;
    if(dataset_name == "Stereo_special") {
        // Make the image
        int max_label = *std::max_element(labeling.begin(), labeling.end());
        int ii = 0;
        for(int i=0; i<estimates.cols(); ++i) {
             if (i == pI[ii]) {
                intensity[2] = 255;
                intensity[1] = 0;
                intensity[0] = 0;
                ++ii;
            } else {
                intensity[2] = 255.0*labeling[i]/max_label;
                intensity[1] = 255.0*labeling[i]/max_label;
                intensity[0] = 255.0*labeling[i]/max_label;
            }

            int col = i % size.width;
            int row = (i - col)/size.width;
            img.at<cv::Vec3b>(row, col) = intensity;
        }
    } else {
        const unsigned char*  legend;
        if (dataset_name == "MSRC") {
            legend = MSRC_legend;
        } else if (dataset_name == "Pascal2010") {
            legend = Pascal_legend;
        } else {
            legend = Stereo_legend;
        }

        // Make the image
        int ii = 0;
        for(int i=0; i<estimates.cols(); ++i) {
            if (i == pI[ii]) {
                intensity[2] = 255;
                intensity[1] = 255;
                intensity[0] = 255;
                ++ii;
            } else {
                intensity[2] = legend[3*labeling[i]];
                intensity[1] = legend[3*labeling[i] + 1];
                intensity[0] = legend[3*labeling[i] + 2];
            }

            int col = i % size.width;
            int row = (i - col)/size.width;
            img.at<cv::Vec3b>(row, col) = intensity;
        }
    }

    cv::imwrite(out_file_name, img);
}

MatrixXf load_matrix(std::string path_to_matrix){
    std::ifstream infile(path_to_matrix.c_str());

    std::string read;
    int nb_rows, nb_cols;

    std::getline(infile, read, '\t');
    nb_rows = stoi(read);
    std::getline(infile, read);
    nb_cols = stoi(read);

    MatrixXf loaded(nb_rows, nb_cols);

    std::string line;
    std::vector<std::string> all_floats;
    for (int i=0; i < nb_rows; i++) {
        std::getline(infile, line);
        split_string(line, ',', all_floats);
        for (int j=0; j< nb_cols; j++) {
            float new_elt = stof(all_floats[j]);
            loaded(i,j) = new_elt;
        }
        all_floats.resize(0);
    }
    return loaded;
}


void save_matrix(std::string path_to_output, MatrixXf matrix){
    std::ofstream file(path_to_output.c_str());
    const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");
    file << matrix.rows() << "\t" << matrix.cols() << std::endl;
    file << matrix.format(CSVFormat);
    file.close();
}
