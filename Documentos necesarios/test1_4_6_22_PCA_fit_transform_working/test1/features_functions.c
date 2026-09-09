#ifdef _features_functions_
#include "features_functions.h"
void matmul(float *A, int A_rows, int A_cols, float *B, int B_rows, int B_cols, float *C){
	int i= 0;
	int j=0;
	int k = 0;
	
	for( i= 0; i<A_rows; i++){
		for(j = 0; j< B_cols; j++){
			C[i*B_cols + j] = 0;
			for( k = 0; k< A_cols; k++){
				C[i*B_cols + j] += A[i*A_cols + k]*B[k*B_cols + j]; 
				}
			
			}
		
		}
	
	}
void matsum(float *A,  float *B, int rows, int cols, float *C){
	int i = 0;
	int j = 0;
	
	for(i = 0; i<rows; i++){
		for (j = 0; j<cols; j ++){
			
			C[i*rows + j] = A[i*rows + j] + B[i*rows + j];
			
			}
		
		}
	
	
	}
void matsub(float *A,  float *B, int rows, int cols, float *C){
	int i = 0;
	int j = 0;
	
	for(i = 0; i<rows; i++){
		for (j = 0; j<cols; j ++){
			
			C[i*rows + j] = A[i*rows + j] - B[i*rows + j];
			
			}
		
		}
	
	
	}
void mattranspose(float *A, int A_rows, int A_cols, float *B){
	int i = 0;
	int j = 0;
	
	for (i = 0; i< A_rows; i++){
		for (j = 0; j < A_cols; j++){
			
			B[j*A_rows + i] = A[i*A_cols + j];
			
			}
		
		}
	
	}
void sub_mean(float *features, int row_features, int col_features, float *mean, float *features_no_mean){
	int i = 0;
	int k = 0;
	
	for (i = 0; i< row_features; i++){
		matsub(features + i*col_features, mean, 1 ,col_features, features_no_mean + i*col_features);
		
		}
	
	
	
	}
void pca_fit_transform(float *features, int row_features, int col_features, float *mean, float *PCA_matrix, int row_PCA, int col_PCA, float *features_transformed){
	// auxiliar vars
	float features_no_mean[row_features][col_features];
	float PCA_mat_T[col_PCA][row_PCA];
	//sub mean
	sub_mean( features, row_features,col_features, mean, (float *)features_no_mean);
	// transpose PCA matrix
	
	mattranspose(PCA_matrix,row_PCA,col_PCA,(float *)PCA_mat_T);
	//multiply matrices for PCA transformation
	matmul((float *) features_no_mean, row_features, col_features, (float *) PCA_mat_T, col_PCA,row_PCA, features_transformed);

	
	
	}
void printmatrix(float *A, int A_rows, int A_cols){
	int i = 0;
	int j = 0;
	printf("{ ");
	for (i = 0; i<A_rows; i++){
		for (j = 0; j<A_cols; j++ ){
			printf("%.21f  ", A[i*A_cols + j]);
			}
			printf("\n");
		}
	printf("}\n");
	
	}

int sign_of_x(float x){
	int sign_x;
	if (x==0){
		sign_x = 0;
		}
	if (x>0){
		sign_x = 1;
		}
	if (x<0){
		sign_x = -1;
		}
	return sign_x;
	}

float abs_of_x(float x){
	float abs_x;
	if (x==0){
		abs_x = 0;
		}
	if (x>0){
		abs_x = x;
		}
	if (x<0){
		abs_x = -1.0*x;
		}
	return abs_x;
	}

int ZC_function(float *data_segment, int data_segment_length){
	int i = 0;
	int ZC_acc = 0;
	
	for (i =0; i<data_segment_length-1;i++){
		
		if(abs_of_x(data_segment[i] - data_segment[i+1])>zc_threshold){
			if(sign_of_x(data_segment[i]*data_segment[i+1])==-1){
				ZC_acc++;
				}
			
			}
		}

	return ZC_acc;
	}

float RMS_function(float *data_segment, int data_segment_length){
	 int i = 0;
	 float RMS_acc = 0.0;
	 for (i=0; i<data_segment_length;i++){
		 
		 RMS_acc += (data_segment[i]*data_segment[i]);
		 
		 }
	RMS_acc = sqrtf(RMS_acc/data_segment_length);
	
	return RMS_acc;
	}
float WL_function(float *data_segment, int data_segment_length){
	int i = 0;
	float WL_acc = 0.0;
	for (i = 1;i<data_segment_length; i++){
		
		WL_acc += abs_of_x(data_segment[i] -data_segment[i-1]);
		
		
		}
	return WL_acc;
	}
int SSC_function(float *data_segment,int data_segment_length){
	int i = 0;
	int SSC_acc = 0, flag_c1 = 0, flag_c2 = 0, flag_c3 = 0, flag_c4 = 0,flag_c1_th = 0, flag_c2_th = 0;
	int condition_comp = 0, condition_threshold = 0, final_condition = 0;
	float abs_sub_p = 0.0;
	float abs_sub_m = 0.0;
	for (i = 1; i<data_segment_length -1; i++){
		abs_sub_p = abs_of_x(data_segment[i] - data_segment[i+1]);
		abs_sub_m = abs_of_x(data_segment[i] - data_segment[i-1]);
		flag_c1 = (data_segment[i]>data_segment[i-1])? 1:0;
		flag_c2 = (data_segment[i]>data_segment[i+1])? 1:0;
		flag_c3 = (data_segment[i]<data_segment[i-1])? 1:0;
		flag_c4 = (data_segment[i]<data_segment[i+1])? 1:0;
		flag_c1_th = (abs_sub_m>ssc_threshold)? 1:0;
		flag_c2_th = (abs_sub_p>ssc_threshold)? 1:0;
		condition_comp = (flag_c1 && flag_c2) || (flag_c3 && flag_c4)? 1:0;
		condition_threshold = (flag_c1_th || flag_c2_th)? 1:0;
		final_condition = (condition_comp && condition_threshold)? 1:0;
		SSC_acc =(final_condition)? SSC_acc+=1:SSC_acc;
		}
	return SSC_acc;
	}
	float MAV_function(float *data_segment,int data_segment_length){
		int i = 0;
		float MAV_acc = 0.0;
		
		for (i=0; i < data_segment_length; i++){
			
			MAV_acc += abs_of_x(data_segment[i]);
			
			}
			MAV_acc = MAV_acc/data_segment_length;
			return MAV_acc;
		}
	float MAVS_function(float *data_segment_1,int data_segment_length_1,float *data_segment_2,int data_segment_length_2){
		float MAVS = 0.0;
		
		MAVS = MAV_function(data_segment_2,data_segment_length_2) - MAV_function(data_segment_1,data_segment_length_1);
		printf("mavs %.21f\n",MAVS);
		return MAVS;
		
		}
#endif