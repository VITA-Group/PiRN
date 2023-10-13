import os

path = "/data/ajay_data/purdue_data"
data_folder = "train_static"

fopen_lq = open("./train_data/train_data_lq.txt","w")
fopen_gt = open("./train_data/train_data_gt.txt","w")
folders = os.listdir(os.path.join(path, data_folder))
for folder in folders:
    gt_path = os.path.join(os.path.join(path, data_folder, folder, "gt.jpg"))
    input_images_folder = os.listdir(os.path.join(path, data_folder, folder, "turb"))
    for image in input_images_folder:
        input_image = os.path.join(path, data_folder, folder, "turb", image)
        fopen_lq.write(f"{input_image}\n")
        fopen_gt.write(f"{gt_path}\n")
        fopen_lq.flush()
        fopen_gt.flush()
fopen_gt.close()
fopen_lq.close()

