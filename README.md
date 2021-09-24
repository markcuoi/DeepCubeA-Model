# DeepCubeA-Model

### Code simulatation: ###
install environments:
python3
vpython : pip3 install vpython
pygame: pip3 install pygame

python main.py hoặc run file trong pycharm #### run code
file output.txt copy vào cùng thư mục với main.py


### deepcubeA ###
environments:
conda pytorch + cuda toolkit
run file: 
python search_methods/astar.py --env cube3 --states data/cube3/handing/samples/test.pkl --model_dir saved_models/cube3/current --results_dir results/cube3 --verbose --batch_size 500 --weight 0.2 --nnet_batch_size 10000

file file test.pkl paste vao data/cube3/handing/samples/
Sau khi giai ra solution chỉ cần chạy file simulate xem kết quả

## nhớ  cd đến thư mục Rubik
