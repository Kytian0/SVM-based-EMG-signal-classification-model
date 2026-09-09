
#include "features_for_movements_0_vs_1.h" 
#include "features_for_movements_0_vs_1.c" 
#include "features_functions.h"
#include "features_functions.c"
#include <string.h>

int main(int argc, char **argv)
{
   // printf() displays the string inside quotation
   printf("Hello, World!\n");
   
   int zc_test = 0;
   float rms_test = 0.0;
   float wl_test = 0.0;
   int ssc_test = 0; 
   float mav_test = 0;
   float mavs_test = 0;
   init_params_mov_0_vs_1();
   printf("feature 0 %d \n",params_mov_0_vs_1.used_features[0]);
   printf("feature 1 %d \n",params_mov_0_vs_1.used_features[1]);
   printf("feature 2 %d \n",params_mov_0_vs_1.used_features[2]);
   printf("PCA 0 1 %f \n", PCA_matrix_mov_0_vs_1[0][1]);
   // Matrix operations

   float A_matrix[3][3] = {{1,2,3},{4,5,6},{7,8,9}};
   int num_rows_A = (int) sizeof(A_matrix) / (int) sizeof(A_matrix[0]);
   int num_cols_A = (int) sizeof(A_matrix[0]) / (int) sizeof(A_matrix[0][0]);
   float B_matrix[3][2] ={{1,2},{1,2},{1,2}};
   int num_rows_B = (int) sizeof(B_matrix) / (int) sizeof(B_matrix[0]);
   int num_cols_B = (int) sizeof(B_matrix[0]) / (int) sizeof(B_matrix[0][0]);
   float C_matrix[num_rows_A][num_cols_B];
   float A_T[num_cols_A][num_rows_A];
   float* pointer = (float*) A_matrix;
   printf("A matrix element: %.21f", pointer[1*num_cols_A + 1]);
   // Matrix multiplication test
   matmul((float *)A_matrix,num_rows_A,num_cols_A,(float *)B_matrix,num_rows_B,num_cols_B,(float *)C_matrix);
   printf("A matrix: \n");
   printmatrix((float *)A_matrix,num_rows_A,num_cols_A);
   printf("B matrix: \n");
   printmatrix((float *)B_matrix,num_rows_B,num_cols_B);
   printf("C matrix: \n");
   printmatrix((float *)C_matrix,num_rows_A,num_cols_B);
   // Matrix transpose test
   
   mattranspose((float *)A_matrix,num_rows_A,num_cols_A,(float *)A_T);
   printf("A transpose: \n");
   printmatrix((float *) A_T, num_cols_A,num_rows_A);
   
   // Matrix addition - substraction test
   float D_matrix[3][3] =  {{2,3,4},{5,6,7},{8,9,10}};
   int num_rows_D = (int) sizeof(D_matrix) / (int) sizeof(D_matrix[0]);
   int num_cols_D = (int) sizeof(D_matrix[0]) / (int) sizeof(D_matrix[0][0]);
   float A_D_add[num_rows_D][num_cols_D];
   float A_D_sub[num_rows_D][num_cols_D];
   printf("D matrix: \n");
   printmatrix((float *) D_matrix,num_rows_D,num_cols_D);
   matsum((float *) A_matrix,(float *)D_matrix, num_rows_D,num_cols_D, (float *) A_D_add);
   printf("A + D matrix: \n");
   printmatrix((float *) A_D_add, num_rows_D, num_cols_D);
   matsub((float *) A_matrix,(float *)D_matrix, num_rows_D,num_cols_D, (float *) A_D_sub);
   printf("A - D matrix: \n");
   printmatrix((float *) A_D_sub, num_rows_D, num_cols_D);
   
   //PCA transform
   float mean[1][3] = {{4,5,6}};
   int num_rows_mean = (int) sizeof(mean) / (int) sizeof(mean[0]);
   int num_cols_mean = (int) sizeof(mean[0]) / (int) sizeof(mean[0][0]);
   float features_to_transform [2][3] = {{42,7,8},{6,7,8}};
   int num_rows_feat = (int) sizeof(features_to_transform) / (int) sizeof(features_to_transform[0]);
   int num_cols_feat = (int) sizeof(features_to_transform[0]) / (int) sizeof(features_to_transform[0][0]);
   float feat_no_mean[num_rows_feat][num_cols_feat];
   sub_mean((float *) features_to_transform, num_rows_feat, num_cols_feat, (float *) mean, (float *) feat_no_mean);
   printf("feat - mean: \n" );
   printmatrix((float *) feat_no_mean, num_rows_feat, num_cols_feat);
   float PCA_mat[2][3] = {{-0.577335027, -0.57735027, -0.57735027},{-0.81649658, 0.40824829, 0.40824829}};
   int num_rows_PCA = (int) sizeof(PCA_mat) / (int) sizeof(PCA_mat[0]);
   int num_cols_PCA = (int) sizeof(PCA_mat[0]) / (int) sizeof(PCA_mat[0][0]);
   float PCA_mat_T[num_cols_PCA][num_rows_PCA];
   mattranspose((float *)PCA_mat,num_rows_PCA,num_cols_PCA,(float *)PCA_mat_T);
   float transformed_features[num_rows_feat][num_rows_PCA];
   matmul((float *) feat_no_mean, num_rows_feat, num_cols_feat, (float *) PCA_mat_T, num_cols_PCA,num_rows_PCA,(float*)transformed_features);
   printf("Transformed features: \n");
   printmatrix((float *) transformed_features,num_rows_feat,num_rows_PCA);
   
   // PCA fit transform function test
   float mean_test[1][3] = {{4,5,6}};
   float features_to_transform_test [2][3] = {{42,7,8},{6,7,8}};
   int num_rows_feat_test = (int) sizeof(features_to_transform_test) / (int) sizeof(features_to_transform_test[0]);
   int num_cols_feat_test = (int) sizeof(features_to_transform_test[0]) / (int) sizeof(features_to_transform_test[0][0]);
   float PCA_mat_test[2][3] = {{-0.577335027, -0.57735027, -0.57735027},{-0.81649658, 0.40824829, 0.40824829}};
   int num_rows_PCA_test = (int) sizeof(PCA_mat_test) / (int) sizeof(PCA_mat_test[0]);
   int num_cols_PCA_test = (int) sizeof(PCA_mat_test[0]) / (int) sizeof(PCA_mat_test[0][0]);
   float transformed_features_test[num_rows_feat_test][num_rows_PCA_test];
   pca_fit_transform((float *)features_to_transform_test, num_rows_feat_test, num_cols_feat_test,
					 (float *)mean_test, (float *)PCA_mat_test, num_rows_PCA_test, num_cols_PCA_test,
					 (float *)transformed_features_test);
   printf("Transformed features via pca_fit_transform function: \n");
   printmatrix((float *) transformed_features_test,num_rows_feat_test,num_rows_PCA_test);
   
   // Feature extraction
   float segment_array[4] ={1.0,2.0,1.0,2.0};
   float segment_array_2[4] = {1.0,2.0,3.0,4.0};
   int segment_array_size = (int) (sizeof(segment_array)/sizeof(segment_array[0]));
   int segment_array_size_2 = (int) (sizeof(segment_array_2)/sizeof(segment_array_2[0]));
   zc_test = ZC_function(&segment_array[0],segment_array_size);
   rms_test = RMS_function(&segment_array[0],segment_array_size);
   wl_test = WL_function(&segment_array[0],segment_array_size);
   ssc_test = SSC_function(&segment_array[0],segment_array_size);
   mav_test = MAV_function(&segment_array[0],segment_array_size);
   mavs_test = MAVS_function(&segment_array[0],segment_array_size,&segment_array_2[0],segment_array_size_2);
   printf("ZC_function test %d \n", zc_test);
   printf("RMS function test %.21f \n", rms_test);
   printf("WL function test %.21f \n", wl_test);
   printf("SSC_function test %d \n", ssc_test);
   printf("MAV function test %.21f \n", mav_test);
   printf("MAVS function test %.21f \n", mavs_test);
    return 0;
}
