学习神经网络，体会

11.28
backward分别实现了
backward_pass_np 使用numpy完成
backward_pass_torch 用tensor代替ndarray
backward_tensor_backward.py 调用backward()
backward_model.py 使用模型，模型调用backward()
backward_model_optim.py 利用optim来优化反向求导
backward_model_optim.py 创建MyModel


RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED 不知道为什么