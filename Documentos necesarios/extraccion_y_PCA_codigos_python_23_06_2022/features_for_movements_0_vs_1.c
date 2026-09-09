#ifdef _FEATURES_FOR_MOVEMENTS_0_vs_1_
#include "features_for_movements_0_vs_1.h" 
void init_params_mov_0_vs_1()
{
params_mov_0_vs_1.used_features[0] = MAV_emg_1_mov_0_vs_1;
params_mov_0_vs_1.used_features[1] = MAV_emg_8_mov_0_vs_1;
params_mov_0_vs_1.used_features[2] = MAV_emg_10_mov_0_vs_1;
params_mov_0_vs_1.used_features[3] = RMS_emg_1_mov_0_vs_1;
params_mov_0_vs_1.used_features[4] = RMS_emg_8_mov_0_vs_1;
params_mov_0_vs_1.used_features[5] = RMS_emg_10_mov_0_vs_1;
}
#endif