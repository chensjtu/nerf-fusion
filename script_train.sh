### for test nerfusion
# conda activate dif

# python example/nerfusion/run_nf.py --config example/nerfusion/configs/scannet_local.txt 

### depth 
# python example/nerfusion/run_nf.py --config example/nerfusion/configs/debug_scannet.txt --expname depth_nerf_scene1 --loss_w_depth 1

### test precedure
# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_scannet.txt --expname test_nerfusion
# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_scannet.txt --expname test_nerfusion_dp --loss_w_depth 1

### export mesh
# CUDA_VISIBLE_DEVICES=1 python example/nerfusion/test_nerfusion.py \
#                                 --config example/nerfusion/configs/test_scannet.txt \
#                                 --expname test_nerfusion_dp \
#                                 --pt_per_edge 3 \
#                                 --export_mesh

# CUDA_VISIBLE_DEVICES=1 python example/nerfusion/test_nerfusion.py \
#                                 --config example/nerfusion/configs/test_scannet.txt \
#                                 --expname test_nerfusion_dp \
#                                 --pt_per_edge 3 \
#                                 --export_mesh \
#                                 --threshold 0

# CUDA_VISIBLE_DEVICES=1 python example/nerfusion/test_nerfusion.py \
#                                 --config example/nerfusion/configs/test_scannet.txt \
#                                 --expname test_nerfusion_dp \
#                                 --pt_per_edge 3 \
#                                 --export_mesh \
#                                 --threshold 1

# CUDA_VISIBLE_DEVICES=1 python example/nerfusion/test_nerfusion.py \
#                                 --config example/nerfusion/configs/test_scannet.txt \
#                                 --expname test_nerfusion_dp \
#                                 --pt_per_edge 3 \
#                                 --export_mesh \
#                                 --threshold 2

# CUDA_VISIBLE_DEVICES=1 python example/nerfusion/test_nerfusion.py \
#                                 --config example/nerfusion/configs/test_scannet.txt \
#                                 --expname test_nerfusion_dp \
#                                 --pt_per_edge 3 \
#                                 --export_mesh \
#                                 --threshold 5

### exp for region loss
# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_scannet.txt --expname region_loss_0_04 
# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_scannet.txt --expname region_loss_0_02 --region_epsilon 0.02
# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_scannet.txt --expname region_loss_0_06 --region_epsilon 0.06
# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_scannet.txt --expname region_loss_0_08 --region_epsilon 0.08
# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_scannet.txt --expname region_loss_0_10 --region_epsilon 0.10 

# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_scannet.txt --expname region_loss_0_04_no_dp --loss_w_depth 0.0
# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_scannet.txt --expname region_loss_0_08_no_dp --loss_w_depth 0.0 --region_epsilon 0.08

# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_scannet.txt --expname region_loss_0_20 --N_iters 100000 --i_print 10 --i_weights 10000 --i_decay_epsilon 500

### fix the bug of depth. this script is load from 040000.pt
# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_scannet.txt \
#                                             --expname region_loss_0_20_only_detach \
#                                             --N_iters 100000 \
#                                             --i_print 10 \
#                                             --N_views 2 \
#                                             --N_rand 16384 \
#                                             --i_weights 10000 \
#                                             --i_decay_epsilon 500 \
#                                             --epsilon_steps 50 \
#                                             --ft_path logs/region_loss_0_20/040000.pt \
#                                             --loss_w_rgb 0.0 \
#                                             --loss_w_depth 0.0

### test the region loss, to see whether it will down 2->0.2
# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_scannet.txt \
#                                             --expname only_region_loss_linear \
#                                             --N_iters 50000 \
#                                             --i_print 100 \
#                                             --N_views 2 \
#                                             --N_rand 10000 \
#                                             --i_weights 1000 \
#                                             --i_decay_epsilon 100 \
#                                             --epsilon_steps 50 \
#                                             --loss_w_rgb 0.0 \
#                                             --loss_w_depth 0.0

# ### another exp, no linear steps. with region_epsilon 1, 1->1
# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_scannet.txt \
#                                             --expname only_region_loss_02_new \
#                                             --N_iters 20000 \
#                                             --i_print 100 \
#                                             --N_views 2 \
#                                             --N_rand 10000 \
#                                             --i_weights 5000 \
#                                             --i_decay_epsilon 100 \
#                                             --epsilon_steps 100 \
#                                             --loss_w_rgb 0.0 \
#                                             --loss_w_depth 0.0 \
#                                             --region_epsilon 0.2 \
#                                             --loss_w_region 100 

# ### another exp, no linear steps. with region_epsilon 0.2, 0.2->0.2
# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_scannet.txt \
#                                             --expname only_region_loss_no_linear02 \
#                                             --N_iters 50000 \
#                                             --i_print 100 \
#                                             --N_views 2 \
#                                             --N_rand 10000 \
#                                             --i_weights 10000 \
#                                             --i_decay_epsilon 100 \
#                                             --epsilon_steps 0 \
#                                             --loss_w_rgb 0.0 \
#                                             --loss_w_depth 0.0 \
#                                             --region_epsilon 0.2

# ### another exp, no linear steps. with region_epsilon 1, rgb loss, 1->1
# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_scannet.txt \
#                                             --expname rgb_no_linear2 \
#                                             --N_iters 50000 \
#                                             --i_print 100 \
#                                             --N_views 2 \
#                                             --N_rand 20000 \
#                                             --i_weights 10000 \
#                                             --i_decay_epsilon 100 \
#                                             --epsilon_steps 0 \
#                                             --loss_w_rgb 32.0 \
#                                             --loss_w_depth 0.0 \
#                                             --region_epsilon 1

### one exp to verify the pure rgb and depth loss
# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_scannet.txt \
#                                             --expname rgb_depth_loss \
#                                             --N_iters 50000 \
#                                             --i_print 100 \
#                                             --N_views 2 \
#                                             --N_rand 20000 \
#                                             --i_weights 10000 \
#                                             --loss_w_rgb 32.0 \
#                                             --loss_w_depth 1.0 \
#                                             --loss_w_region 0.0 \
#                                             --loss_w_unregion 0.0


### use blender dataset, rgb & depth loss
# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_blender.txt \
#                                             --expname blender_base_line \
#                                             --loss_w_depth 1.0 \
#                                             --loss_w_rgb 1.0 \
#                                             --loss_w_region 0.0 \
#                                             --loss_w_unregion 0.0 \
#                                             --voxel_size 0.02 \
#                                             --N_iters 20000 \
#                                             --N_rand 12000 \
#                                             --N_views 2

### use local coord as input
# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_blender.txt \
#                                             --expname test_blender_1 \
#                                             --loss_w_depth 1.0 \
#                                             --loss_w_rgb 1.0 \
#                                             --loss_w_region 0.0 \
#                                             --loss_w_unregion 0.0 \
#                                             --voxel_size 0.02 \
#                                             --N_iters 20000 \
#                                             --N_rand 10000 \
#                                             --N_views 2

### use new region loss
# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_blender.txt \
#                                             --expname blender_1_region_linear \
#                                             --loss_w_depth 1.0 \
#                                             --loss_w_rgb 1.0 \
#                                             --loss_w_region 100.0 \
#                                             --loss_w_unregion 1000.0 \
#                                             --voxel_size 0.02 \
#                                             --N_iters 20000 \
#                                             --N_rand 12000 \
#                                             --N_views 2

# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_blender.txt \
#                                             --expname blender_1_region_nolinear1 \
#                                             --loss_w_depth 1.0 \
#                                             --loss_w_rgb 1.0 \
#                                             --loss_w_region 100.0 \
#                                             --loss_w_unregion 1000.0 \
#                                             --voxel_size 0.02 \
#                                             --N_iters 20000 \
#                                             --N_rand 12000 \
#                                             --N_views 2 \
#                                             --epsilon_steps 0 \
#                                             --region_epsilon 1

# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_blender.txt \
#                                             --expname blender_1_region_nolinear02 \
#                                             --loss_w_depth 1.0 \
#                                             --loss_w_rgb 1.0 \
#                                             --loss_w_region 100.0 \
#                                             --loss_w_unregion 1000.0 \
#                                             --voxel_size 0.02 \
#                                             --N_iters 20000 \
#                                             --N_rand 12000 \
#                                             --N_views 2 \
#                                             --epsilon_steps 0 \
#                                             --region_epsilon 0.2

python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_blender.txt \
                                            --expname all_frames_train \
                                            --loss_w_depth 1.0 \
                                            --loss_w_rgb 1.0 \
                                            --loss_w_region 0.0 \
                                            --loss_w_unregion 0.0 \
                                            --loss_w_alpha 0 \
                                            --voxel_size 0.02 \
                                            --N_iters 50000 \
                                            --N_rand 2048 \
                                            --N_views 1 \
                                            --export_mesh \
                                            --pt_per_edge 3

# python example/nerfusion/test_nerfusion.py --config example/nerfusion/configs/test_blender.txt \
#                                             --expname blender_region_linear_alpha \
#                                             --loss_w_depth 0.0 \
#                                             --loss_w_rgb 0.0 \
#                                             --loss_w_region 100.0 \
#                                             --loss_w_unregion 1000.0 \
#                                             --loss_w_alpha 1 \
#                                             --voxel_size 0.02 \
#                                             --N_iters 20000 \
#                                             --N_rand 8192 \
#                                             --N_views 1