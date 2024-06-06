cd graf-main

python render_xray_G.py configs/knee.yaml --xray_img_path /root/yf/mednerf/data/knee_test --save_dir ./output  --model /root/yf/mednerf/checkpoints/model_best_knee.pt --save_every 25 --psnr_stop 25 