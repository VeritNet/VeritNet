/*
 * 转自 https ://www.cnblogs.com/ppDoo/p/13261258.html
 * 原作者版权所有
 * 仅用于测试使用
*/

#pragma once
#include <iostream>  
#include <fstream> 
#include <vector> 


using namespace std;

float(*train_image)[784] = new float[50000][784];
float(*train_label)[10] = new float[50000][10];

inline int ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

inline void read_Mnist_Label(char filename[], vector<int>& labels, int total)
{
    ifstream file(filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);

        for (int i = 0; i < total; i++)
        {
            unsigned char label = 0;
            file.read((char*)&label, sizeof(label));
            labels.push_back((double)label);
        }
    }
}

inline void read_Mnist_Images(char filename[], vector< vector<int> >& images, int total)
{
    ifstream file(filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        unsigned char label;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        file.read((char*)&n_rows, sizeof(n_rows));
        file.read((char*)&n_cols, sizeof(n_cols));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        n_rows = ReverseInt(n_rows);
        n_cols = ReverseInt(n_cols);

        for (int i = 0; i < total; i++)
        {
            vector<int>tp;
            for (int r = 0; r < n_rows; r++)
            {
                for (int c = 0; c < n_cols; c++)
                {
                    unsigned char image = 0;
                    file.read((char*)&image, sizeof(image));
                    tp.push_back(image);
                }
            }
            images.push_back(tp);
            if (i % 6000 == 0) {
                std::cout << i << "/" << total << " Loaded" << std::endl;
            }
        }
    }
}

inline void convert_array_image(vector< vector<int> >& feature_vector)
{
    for (int i = 0; i < feature_vector.size(); i++)
    {
        for (int a = 0; a < 28; a++)
        {
            for (int b = 0; b < 28; b++)
            {
                train_image[i][a*28+b] = feature_vector[i][a * 28 + b] / static_cast<float>(255);
            }
        }
    }
}

inline void convert_array_label(vector<int>& labels)
{
    for (int i = 0; i < labels.size(); i++)
    {
        train_label[i][labels[i]] = 1;
    }
}