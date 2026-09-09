#include <stdio.h>
#include "features_for_movements_0_vs_1.h" 
#include "features_for_movements_0_vs_1.c" 
int main() {
   // printf() displays the string inside quotation
   printf("Hello, World!\n");
  
   init_params_mov_0_vs_1();
   printf("feature 0 %d \n",params_mov_0_vs_1.used_features[0]);
   printf("feature 1 %d \n",params_mov_0_vs_1.used_features[1]);
   printf("feature 2 %d \n",params_mov_0_vs_1.used_features[2]);
   printf("PCA 0 1 %f \n", PCA_matrix_mov_0_vs_1[0][1]);
    return 0;
}
