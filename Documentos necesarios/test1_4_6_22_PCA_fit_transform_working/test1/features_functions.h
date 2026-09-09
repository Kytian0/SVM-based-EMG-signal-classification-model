#ifndef _features_functions_
#define _features_functions_
#include <stdint.h>
#include <math.h>
#include <stdio.h>
float zc_threshold = 1e-5;
float ssc_threshold = 1e-5;

int sign_of_x(float x);
float abs_of_x(float x);
void matmul(float *A, int A_rows, int A_cols, float *B, int B_rows, int B_cols, float *C);
void matsum(float *A,  float *B, int rows, int cols, float *C);
void matsub(float *A,  float *B, int rows, int cols, float *C);
void mattranspose(float *A, int A_rows, int A_cols, float *B);
void sub_mean(float *features, int row_features, int col_features, float *mean, float *features_no_mean);
void pca_fit_transform(float *features, int row_features, int col_features, float *mean, float *PCA_matrix, int row_PCA, int col_PCA, float *features_transformed);
void printmatrix(float *A, int A_rows, int A_cols);
int ZC_function(float *data_segment, int data_segment_length);
float RMS_function(float *data_segment, int data_segment_length);
float WL_function(float *data_segment, int data_segment_length);
int SSC_function(float *data_segment,int data_segment_length);
float MAV_function(float *data_segment,int data_segment_length);
float MAVS_function(float *data_segment_1,int data_segment_length_1,float *data_segment_2,int data_segment_length_2);
#endif